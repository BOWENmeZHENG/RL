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
import rl
import pandas as pd

def do_training(random_samples: bool, prompt: list, fixed_ids: list, mean_random, std_random,
                episodes, d_embed, learning_rate, scheduler, scheduler_period, vocabulary, hidden, seed, 
                print_interval, save_results, plot, client, dataset, dataset_test,
                format_prompt=". Format it strictly as entities separated by comma.", model_name="gpt-4o-mini",
                smallest_lr=1e-6, decay_rate=0.5, plot_interval=50):
    #--------- Setup ---------------------
    utils.seed_everything(seed)                
    data = utils.load_json_file(f"data/{dataset}.json")
    if dataset_test is not None:
        data_t = utils.load_json_file(f"data/{dataset_test}.json")
        print("Test dataset loaded.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vocab_words = pd.read_csv(f"vocabulary/{vocabulary}.txt").columns.tolist()
    print(f"Vocabulary size: {len(vocab_words)}")
    if random_samples:
        exp_name = f"RANDOM_d_{dataset}_p_{prompt}_ids_{fixed_ids}_v_{vocabulary}_e_{episodes}_s_{seed}"
    else:
        exp_name = f"d_{dataset}_dt_{dataset_test}_p_{prompt}_ids_{fixed_ids}_stats_{mean_random}_{std_random}_v_{vocabulary}_e_{episodes}_l_{learning_rate}_sch_{scheduler}_schp_{scheduler_period}_h_{hidden}_s_{seed}"
    
    #-------------- Initialization (the very first prompt designed by human) --------------------------
    prompt_str = " ".join(prompt)
    embedding = torch.tensor(utils.get_embedding(prompt_str, client, d_embed)).to(device)
    
    # Convert list of words to a sentence and add format prompt
    prompt_complete = prompt_str + format_prompt
    
    # ************************************** Training *******************************************************
    
    # ------------------Define networks -----------------------------------------------
    """
    Action prob length: N * V
    
    Location 1: Word 1, Word 2 ... Word V
    Location 2: Word 1, Word 2 ... Word V
    ...
    Location N: Word 1, Word 2 ... Word V
    
    Split and rescale into N probability vectors
    
    Sample N actions
    """
    n_actions = len(prompt) - 1
    net = rl.PolicyNet(input_dim=d_embed, hidden_dim=hidden, output_dim=len(vocab_words)*n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if scheduler:
        lr_func = lambda itr: decay_rate ** itr
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    
    # ------------------- Calculate predictions and reward of the first prompt ----------------------------
    PROMPTS, PREDICTIONS, SCORES, REWARDS = [], [], [], []
    PREDICTIONS_T, SCORES_T, REWARDS_T = [], [], []
    
    scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
    print(f"Initial prompt: {prompt}")
    print(f"Reward: {reward}\n")
    SCORES.append(scores)
    REWARDS.append(reward)
    PREDICTIONS.append(predictions)
    PROMPTS.append(prompt)
    if dataset_test is not None:
        scores_t, predictions_t, reward_t = rl.reward_function(prompt_complete, data_t, client)
        print(f"Reward test: {reward_t}\n")
        SCORES_T.append(scores_t)
        REWARDS_T.append(reward_t)
        PREDICTIONS_T.append(predictions_t)
    
    # --------------------------- Training loops ---------------------------------
    best_prompt = prompt.copy()
    best_reward = reward
    if dataset_test is not None:
        best_prompt_t = prompt.copy()
        best_reward_t = 0
    
    for episode in range(episodes):
        lr_current = optimizer.param_groups[0]['lr']
        # Location
        if random_samples:
            actions = random.sample(list(range(len(vocab_words))), n_actions)
        else:
            action_probs_all = net.forward(embedding).reshape(n_actions, -1) # N by V
            action_probs_all_n = action_probs_all / action_probs_all.sum(dim=1, keepdim=True)
            actions, log_probs = [], []
            sum_log_prob = 0
            for action_probs in action_probs_all_n:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                sum_log_prob += log_prob
                actions.append(action)
                log_probs.append(log_prob)

        counter = 0
        for i in range(len(prompt)):
            if i not in fixed_ids:
                prompt[i] = vocab_words[actions[counter]]
                counter += 1
        prompt_str = " ".join(prompt)
        prompt_complete = prompt_str + format_prompt
        
        # calculate reward
        scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
        
        if not random_samples:
            reward_n = (reward - mean_random) / std_random
            loss = -reward_n * sum_log_prob
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # New state
            embedding = torch.tensor(utils.get_embedding(prompt, client, d_embed)).to(device)
        
            if episode % scheduler_period == 0 and episode != 0 and lr_current > smallest_lr:
                scheduler.step()
                print(f"Current learning rate: {lr_current}")
                
        if reward > best_reward:
            best_prompt = prompt.copy()
            best_reward = reward
        
        if print_interval != 0 and episode % print_interval == 0:
            print(f"Episode: {episode}")
            print(f"Prompt: {prompt} Reward: {reward}")
            print(f"Best prompt: {best_prompt} Best reward: {best_reward}")
            
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt.copy())
        
        if dataset_test is not None:
            scores_t, predictions_t, reward_t = rl.reward_function(prompt_complete, data_t, client)
            if reward_t > best_reward_t:
                best_prompt_t = prompt.copy()
                best_reward_t = reward_t
            if print_interval != 0 and episode % print_interval == 0:
                print(f"Reward test: {reward_t}")
                print(f"Best prompt test: {best_prompt_t} Best reward test: {best_reward_t}")
            SCORES_T.append(scores_t)
            REWARDS_T.append(reward_t)
            PREDICTIONS_T.append(predictions_t)
        
        if plot:
            if (episode + 1) % plot_interval == 0:
                plt.plot(REWARDS, color='g', label="training")
                plt.axhline(y=mean_random, color='r', linestyle='-')
                plt.fill_between(x=[0, episodes], y1=mean_random-std_random, y2=mean_random+std_random, color='r', alpha=.35)
                if dataset_test is not None:
                    plt.plot(REWARDS_T, color='blue', label="test")
                plt.xlim([0, episode + 1])
                plt.xlabel("Episode")
                plt.ylabel("Mean F1 score")
                plt.legend()
                plt.title("Rewards over Episodes")
                plt.show()
    
    if random_samples:
        REWARDS_MEAN, REWARDS_STD = np.mean(REWARDS), np.std(REWARDS)
        print("Reward mean: ", REWARDS_MEAN)
        print("Reward std: ", REWARDS_STD)
    
    if save_results:
        header = ['Prompt', 'Predictions', 'Scores', 'Reward', 'Predictions_test', 'Scores_test', 'Reward_test']
        with open(f'results/{exp_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header) 
            for j in range(len(PROMPTS)):
                writer.writerow([PROMPTS[j], ','.join(map(str, PREDICTIONS[j])), ','.join(map(str, SCORES[j])), REWARDS[j], ','.join(map(str, PREDICTIONS_T[j])), ','.join(map(str, SCORES_T[j])), REWARDS_T[j]])
        
    return PROMPTS, PREDICTIONS, SCORES, REWARDS