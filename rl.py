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
        return probs

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
    
def do_training(prompt_init, epochs, learning_rate, vocal_size, prompt_length, hidden,  
                exp_id, print_interval, save_results, plot, client, 
                dataset="train_data.json", pad_token_id=220, seed=1234,
                format_prompt="Format it strictly as entities separated by comma.", model_name="gpt-4o"):
    utils.seed_everything(seed)                
    data = utils.load_json_file(dataset)
    # Initialization
    encoding = tiktoken.encoding_for_model(model_name)
    tokens_prompt = encoding.encode(prompt_init)
    padded_token_ids = tokens_prompt + [pad_token_id] * (prompt_length - len(tokens_prompt))
    token_tensor = torch.tensor(padded_token_ids, dtype=torch.float)
    token_tensor = torch.reshape(token_tensor, (1, -1))
    scaler = MinMaxScaler()
    scaler.fit(token_tensor)
    token_tensor = torch.from_numpy(scaler.transform(token_tensor)).float()
    prompt_complete = prompt_init + format_prompt
    
    policy_network = PolicyNet(input_dim=prompt_length, hidden_dim=hidden, output_dim=prompt_length)
    optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
    
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
        action_probs = policy_network.forward(token_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    
        # calculate reward
        token_original = scaler.inverse_transform(token_tensor.detach().numpy()).astype(int)
        token_original = token_original.clip(min=0, max=vocal_size)
        prompt_original = encoding.decode(token_original[0].tolist())
        token_original[0][action.item()] = random.randint(0, vocal_size)
        prompt = encoding.decode(token_original[0].tolist())
        prompt_complete = prompt + format_prompt
        scores, predictions, reward = reward_function(prompt_complete, data, client)
        if print_interval != 0 and episode % print_interval == 0:
            print(episode, prompt)
            print(f"Reward: {reward}")
    
        loss = -(reward - 0.5) * 2 * log_prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt)
    
        # New state
        token_tensor = torch.from_numpy(scaler.transform(token_original)).float()
    
    if plot:
        plt.plot(REWARDS)
        plt.xlabel("Episode")
        plt.ylabel("Mean F1 score")
        plt.title("Rewards over Episodes")
        plt.show()
    
    if save_results:
        exp_name = f"{prompt_init}_e_{epochs}_l_{learning_rate}_v_{vocal_size}_len_{prompt_length}_h_{hidden}_s_{seed}_id_{exp_id}"
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