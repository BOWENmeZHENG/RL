import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import random
import utils
import tiktoken
from sklearn.preprocessing import MinMaxScaler
import csv
import matplotlib.pyplot as plt




class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = torch.softmax(logits, dim=-1)  # action probabilities
        return probs.clone()

def reward_function(prompt_complete, data, client):
    scores, predictions = [], []
    for entry in data:
        text = entry['paragraph']
        truth = entry['materials']
        response = utils.extract_named_entities(prompt_complete, text, client, temperature=0.0, max_tokens=100)
        prediction = response.split(',')
        score = utils.calculate_precision_recall_with_iou(prediction, truth, iou_threshold=0.5)[-1] # F1 score
        scores.append(score)
        predictions.append(prediction)
    reward = np.mean(scores)
    return scores, predictions, reward


def do_training(prompt_init: list, fixed_ids: list, epochs, learning_rates, v_size, word_len_min, hiddens, seed,
                exp_id, print_interval, save_results, plot, client, dataset, 
                pad_token_id=220, format_prompt=". Format it strictly as entities separated by comma.", model_name="gpt-4o-mini"):
    utils.seed_everything(seed)                
    data = utils.load_json_file(f"data/{dataset}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialization
    encoding = tiktoken.encoding_for_model(model_name)
    vocab = utils.make_english_vocab(encoding, v_size, word_len_min)
    print(f"Vocabulary size: {len(vocab)}")
    tokens_prompt = [encoding.encode(word)[0] for word in prompt_init] # encoding.encode(prompt_init)
    # padded_token_ids = tokens_prompt + [pad_token_id] * (prompt_length - len(tokens_prompt))
    token_tensor = torch.tensor(tokens_prompt, dtype=torch.float)
    token_tensor = torch.reshape(token_tensor, (-1, 1))
    scaler = MinMaxScaler()
    scaler.fit(token_tensor)
    token_tensor_normalized = torch.from_numpy(scaler.transform(token_tensor)).float()
    token_tensor_normalized = torch.reshape(token_tensor_normalized, (1, -1))
    prompt_str = " ".join(prompt_init)
    prompt_complete = prompt_str + format_prompt
    
    prompt_length = len(prompt_init)
    net_loc = PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[0], output_dim=prompt_length).to(device)
    optimizer_loc = optim.Adam(net_loc.parameters(), lr=learning_rates[0])
    net_word = PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[1], output_dim=len(vocab)).to(device)
    optimizer_word = optim.Adam(net_word.parameters(), lr=learning_rates[1])
    
    PROMPTS, PREDICTIONS, SCORES, REWARDS = [], [], [], []
    scores, predictions, reward = reward_function(prompt_complete, data, client)
    print(f"Initial prompt: {prompt_init}")
    print(f"Predictions: {predictions}")
    print(f"Scores: {scores}")
    print(f"Reward: {reward}\n")
    
    SCORES.append(scores)
    REWARDS.append(reward)
    PREDICTIONS.append(predictions)
    PROMPTS.append(prompt_init)
    
    for episode in range(epochs):
        token_tensor_normalized = token_tensor_normalized.to(device)
        action_loc_probs = net_loc.forward(token_tensor_normalized)
        for id in fixed_ids:
            action_loc_probs[0, id] = 0
        action_loc_probs = action_loc_probs / torch.sum(action_loc_probs)
        dist_loc = torch.distributions.Categorical(action_loc_probs)
        action_loc = dist_loc.sample()
        log_prob_loc = dist_loc.log_prob(action_loc)
        
        action_word_probs = net_word.forward(token_tensor_normalized)
        dist_word = torch.distributions.Categorical(action_word_probs)
        action_word = dist_word.sample()
        log_prob_word = dist_word.log_prob(action_word)
        # calculate reward
        token_tensor[action_loc.item(), 0] = vocab[action_word]  # random.choice(vocab)
        print(token_tensor.transpose(0, 1))
        prompt = " ".join([encoding.decode([token]) for token in token_tensor.transpose(0, 1)[0].int().tolist()])
        prompt_complete = prompt + format_prompt
        scores, predictions, reward = reward_function(prompt_complete, data, client)
        if print_interval != 0 and episode % print_interval == 0:
            print(episode, prompt)
            print(f"Predictions: {predictions}")
            print(f"Reward: {reward}")
    
        loss_loc = -(reward - 0.5) * 2 * log_prob_loc
        optimizer_loc.zero_grad()
        loss_loc.backward()
        optimizer_loc.step()
        
        loss_word = -(reward - 0.5) * 2 * log_prob_word
        optimizer_word.zero_grad()
        loss_word.backward()
        optimizer_word.step()
    
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt)
    
        # New state
        token_tensor_normalized = torch.from_numpy(scaler.transform(token_tensor)).float()
        token_tensor_normalized = torch.reshape(token_tensor_normalized, (1, -1))
    
    if plot:
        plt.plot(REWARDS)
        plt.xlabel("Episode")
        plt.ylabel("Mean F1 score")
        plt.title("Rewards over Episodes")
        plt.show()
    
    if save_results:
        exp_name = f"d_{dataset}_p_{prompt_init}_e_{epochs}_l_{learning_rates}_v_{v_size}_len_{prompt_length}_wl_{word_len_min}_h_{hiddens}_s_{seed}_id_{exp_id}"
        header = ['Prompt', 'Predictions', 'Scores', 'Reward']
        with open(f'results/{exp_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header row
    
            for i in range(len(PROMPTS)):
                prompt = PROMPTS[i]
                predictions = ','.join(map(str, PREDICTIONS[i]))  # Convert predictions list to comma-separated string
                scores = ','.join(map(str, SCORES[i])) # Convert scores list to comma-separated string
                reward = REWARDS[i]
    
                writer.writerow([prompt, predictions, scores, reward])
        
    return PROMPTS, PREDICTIONS, SCORES, REWARDS
    
def do_test(prompt, save_results, client, dataset,
            format_prompt="Format it strictly as entities separated by comma.", model_name="gpt-4o-mini"):
    data = utils.load_json_file(f"data/{dataset}.json")
    prompt_complete = prompt + format_prompt
    scores, predictions, reward = reward_function(prompt_complete, data, client)
    print(f"Tested prompt: {prompt}")
    print(f"Predictions: {predictions}")
    print(f"Scores: {scores}")
    print(f"Reward: {reward}\n")
    if save_results:
        exp_name = f"test_{dataset}_p_{prompt}"
        header = ['Prompt', 'Predictions', 'Scores', 'Reward']
        with open(f'results/{exp_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header row
            writer.writerow([prompt, ','.join(map(str, predictions)), scores, reward])
    return None