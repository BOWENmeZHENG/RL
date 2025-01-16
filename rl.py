import torch
import torch.nn as nn
import torch.nn.functional as F
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
from transformers import AutoModel, AutoTokenizer

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
