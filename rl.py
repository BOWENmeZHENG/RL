import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import utils
import tiktoken
import csv
import matplotlib.pyplot as plt
import pickle
from collections import deque
import os



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
        
class ReplayBuffer:
    """
    trainsition: [state, action_loc, log_prob_loc, action_word, log_prob_word, reward]
    
    Example:
    
    [
    tensor([[  6555.],
        [174899.],
        [ 19849.],
        [   220.],
        [  5293.],
        [   220.]]), # state
    4, # action_loc
    tensor([-0.9165], grad_fn=<SqueezeBackward1>), # log_prob_loc
    tensor([447]), # action_word
    tensor([-7.7741], grad_fn=<SqueezeBackward1>), # log_prob_word
    0.6814814814814815 # reward
    ]
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        """Add a transition to the replay buffer."""
        self.buffer.append(transition)
    
    def sample(self):
        """Sample a batch of transitions."""
        state, action_loc, log_prob_loc, action_word, log_prob_word, reward = random.choice(self.buffer)
        return state, action_loc, log_prob_loc, action_word, log_prob_word, reward
        
    def size(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def save(self, file_path):
        """Save the replay buffer to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, file_path):
        """Load the replay buffer from a file."""
        with open(file_path, 'rb') as f:
            self.buffer = pickle.load(f)

class ReplayBufferDataset(Dataset):
    def __init__(self, buffer):
        self.state = [t[0] for t in buffer]
        self.action_loc = [t[1] for t in buffer]
        self.log_prob_loc = [t[2] for t in buffer]
        self.action_word = [t[3] for t in buffer]
        self.log_prob_word = [t[4] for t in buffer]
        self.reward = [t[5] for t in buffer]

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return self.state[idx], self.action_loc[idx], self.log_prob_loc[idx], self.action_word[idx], self.log_prob_word[idx], self.reward[idx]

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


def random_samples(prompt_init: list, fixed_ids: list, epochs, v_size, word_len_min,
                   print_interval, plot, client, dataset,
                   format_prompt=". Format it strictly as entities separated by comma.", model_name="gpt-4o-mini"):
    data = utils.load_json_file(f"data/{dataset}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoding = tiktoken.encoding_for_model(model_name)
    vocab = utils.make_english_vocab(encoding, v_size, word_len_min)
    print(f"Vocabulary size: {len(vocab)}")
    locs_all = list(range(len(prompt_init)))
    locs_avail = [loc for loc in locs_all if loc not in fixed_ids]
    PROMPTS, PREDICTIONS, SCORES, REWARDS = [], [], [], []
    
    # Initial prompt
    prompt_str = " ".join(prompt_init)
    prompt_complete = prompt_str + format_prompt
    tokens_prompt = [encoding.encode(word)[0] for word in prompt_init]
    scores, predictions, reward = reward_function(prompt_complete, data, client)
    SCORES.append(scores)
    REWARDS.append(reward)
    PREDICTIONS.append(predictions)
    PROMPTS.append(prompt_init)
    
    # Following prompts
    for episode in range(epochs):
        # Select a random location
        action_loc = random.choice(locs_avail)
        # Select a random word
        tokens_prompt[action_loc] = random.choice(vocab)
        # Updated prompt
        prompt = " ".join([encoding.decode([token]) for token in tokens_prompt])
        prompt_complete = prompt + format_prompt
        # calculate reward
        scores, predictions, reward = reward_function(prompt_complete, data, client)
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt_init)
        if print_interval != 0 and episode % print_interval == 0:
            print(episode, prompt)
            print(f"Reward: {reward}")
            
    if plot:
        plt.plot(REWARDS)
        plt.xlabel("Episode")
        plt.ylabel("Mean F1 score")
        plt.title("Rewards over Episodes")
        plt.show()
    
    REWARDS_MEAN, REWARDS_STD = np.mean(REWARDS), np.std(REWARDS)
    print("Reward mean: ", REWARDS_MEAN)
    print("Reward std: ", REWARDS_STD)
    
    return PROMPTS, PREDICTIONS, SCORES, REWARDS, REWARDS_MEAN, REWARDS_STD

def do_training(prompt_init: list, fixed_ids: list, add_to_replay, existing_model, mean_random, std_random,
                epochs, learning_rates, v_size, word_len_min, hiddens, seed,
                exp_id, print_interval, save_results, plot, client, dataset, 
                format_prompt=". Format it strictly as entities separated by comma.", model_name="gpt-4o-mini"):
    #--------- Setup ---------------------
    utils.seed_everything(seed)                
    data = utils.load_json_file(f"data/{dataset}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoding = tiktoken.encoding_for_model(model_name)
    vocab = utils.make_english_vocab(encoding, v_size, word_len_min)
    print(f"Vocabulary size: {len(vocab)}")
    exp_name = f"d_{dataset}_p_{prompt_init}_ids_{fixed_ids}_stats_{mean_random}_{std_random}_e_{epochs}_l_{learning_rates}_v_{v_size}_wl_{word_len_min}_h_{hiddens}_s_{seed}_id_{exp_id}"
    
    if add_to_replay:
        replay_name = f"d_{dataset}_p_{prompt_init}_ids_{fixed_ids}_stats_{mean_random}_{std_random}_v_{v_size}_wl_{word_len_min}"
        replay_buffer = ReplayBuffer(capacity=10000)
        try:
            replay_buffer.load(f"buffer/{replay_name}.pkl")
            print(f"Replay buffer {replay_name}.pkl loaded.")
        except:
            replay_buffer.save(f"buffer/{replay_name}.pkl")
            print(f"Create new buffer {replay_name}.pkl")
    
    #-------------- Initialization (the very first prompt designed by human) --------------------------
    tokens_prompt = [encoding.encode(word)[0] for word in prompt_init] # Convert list of words to list of token IDs
    token_tensor = torch.tensor(tokens_prompt, dtype=torch.float)
    token_tensor_normalized = token_tensor / v_size
    
    # Convert list of words to a sentence and add format prompt
    prompt_str = " ".join(prompt_init)
    prompt_complete = prompt_str + format_prompt
    
    # ************************************** Training *******************************************************
    
    # ------------------Define networks -----------------------------------------------
    prompt_length = len(prompt_init)
    net_loc = PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[0], output_dim=prompt_length).to(device)
    optimizer_loc = optim.Adam(net_loc.parameters(), lr=learning_rates[0])
    net_word = PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[1], output_dim=len(vocab)).to(device)
    optimizer_word = optim.Adam(net_word.parameters(), lr=learning_rates[1])
    
    if existing_model is not None:
        net_loc.load_state_dict(torch.load(f'saved_models/{existing_model}/net_loc.pth'))
        net_word.load_state_dict(torch.load(f'saved_models/{existing_model}/net_word.pth'))
    
    # ------------------- Calculate predictions and reward of the first prompt ----------------------------
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
    
    # --------------------------- Training loops ---------------------------------
    for episode in range(epochs):
        transition = []
        transition.append(token_tensor)
        token_tensor_normalized = token_tensor_normalized.to(device)
        # Location
        action_loc_probs = net_loc.forward(token_tensor_normalized)
        for id in fixed_ids:
            action_loc_probs[id] = 1e-10
        action_loc_probs = action_loc_probs / torch.sum(action_loc_probs)
        dist_loc = torch.distributions.Categorical(action_loc_probs)
        action_loc = dist_loc.sample()
        log_prob_loc = dist_loc.log_prob(action_loc)
        # Word
        action_word_probs = net_word.forward(token_tensor_normalized)
        dist_word = torch.distributions.Categorical(action_word_probs)
        action_word = dist_word.sample()
        log_prob_word = dist_word.log_prob(action_word)
        transition.append(action_loc.item())
        transition.append(log_prob_loc)
        transition.append(action_word)
        transition.append(log_prob_word)
        token_tensor[action_loc.item()] = vocab[action_word]

        prompt = " ".join([encoding.decode([token]) for token in token_tensor.int().tolist()])
        prompt_complete = prompt + format_prompt
        
        # calculate reward
        scores, predictions, reward = reward_function(prompt_complete, data, client)
        transition.append(reward)
        if print_interval != 0 and episode % print_interval == 0:
            print(episode, prompt)
            print(f"Reward: {reward}")
        
        reward_n = (reward - mean_random) / std_random
        loss_loc = -reward_n * log_prob_loc
        optimizer_loc.zero_grad()
        loss_loc.backward()
        optimizer_loc.step()
        
        loss_word = -reward_n * log_prob_word
        optimizer_word.zero_grad()
        loss_word.backward()
        optimizer_word.step()
    
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt)
    
        # New state
        token_tensor_normalized = token_tensor / v_size
        if add_to_replay:
            replay_buffer.add(transition)
            replay_buffer.save(f"buffer/{replay_name}.pkl")
    
    if plot:
        plt.plot(REWARDS)
        plt.axhline(y=mean_random, color='r', linestyle='-')
        plt.fill_between(x=[0, epochs], y1=mean_random-std_random, y2=mean_random+std_random, color='r', alpha=.35)
        plt.xlim([0, epochs])
        plt.xlabel("Episode")
        plt.ylabel("Mean F1 score")
        plt.title("Rewards over Episodes")
        plt.show()
    
    if save_results:
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
    
def do_training_exp_rply(prompt_length, replay_name, batch_size, mean_random, std_random, 
                         epochs, learning_rates, v_size, word_len_min, hiddens, seed,
                         print_interval, save_model=True, plot_loss=True, model_name="gpt-4o-mini"):
    #--------- Setup ---------------------
    utils.seed_everything(seed)                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoding = tiktoken.encoding_for_model(model_name)
    vocab = utils.make_english_vocab(encoding, v_size, word_len_min)
    print(f"Vocabulary size: {len(vocab)}")
    model_name = f"h_{hiddens}_rply_{replay_name}_e_{epochs}"
    
    # ------------------Define networks -----------------------------------------------
    net_loc = PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[0], output_dim=prompt_length).to(device)
    optimizer_loc = optim.Adam(net_loc.parameters(), lr=learning_rates[0])
    net_word = PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[1], output_dim=len(vocab)).to(device)
    optimizer_word = optim.Adam(net_word.parameters(), lr=learning_rates[1])
    
    #--------- Load replay buffer ---------------------
    with open(f"buffer/{replay_name}.pkl", 'rb') as f:
        replay_buffer = pickle.load(f)
    print(f"Replay buffer {replay_name}.pkl loaded.")
    
    dataset = ReplayBufferDataset(replay_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # ************************************** Training *******************************************************
    
    losses_loc, losses_word = [], []
    
    # --------------------------- Training loops ---------------------------------
    for episode in range(epochs):
        for batch in dataloader:
            state, action_loc, log_prob_loc, action_word, log_prob_word, reward = batch
            reward_n = (reward - mean_random) / std_random
            loss_loc = torch.mean(-reward_n * log_prob_loc)
            losses_loc.append(loss_loc.detach())
            optimizer_loc.zero_grad()
            loss_loc.backward()
            optimizer_loc.step()
            loss_word = torch.mean(-reward_n * log_prob_word)
            losses_word.append(loss_word.detach())
            optimizer_word.zero_grad()
            loss_word.backward()
            optimizer_word.step()
        if episode % print_interval == 0:
            print(f"Episode {episode + 1}/{epochs}, loss_loc: {loss_loc.item():.4f}, loss_word: {loss_word.item():.4f}")
            
    if plot_loss:
        plt.plot(losses_loc)
        plt.xlabel("Episode")
        plt.ylabel("losses_loc")
        plt.title("losses_loc")
        plt.show()
        
        plt.plot(losses_word)
        plt.xlabel("Episode")
        plt.ylabel("losses_word")
        plt.title("losses_word")
        plt.show()
    
    if save_model:
        model_path = f'saved_models/{model_name}'
        os.makedirs(model_path, exist_ok=True)
        torch.save(net_loc.state_dict(), f'{model_path}/net_loc.pth')
        torch.save(net_word.state_dict(), f'{model_path}/net_word.pth') 