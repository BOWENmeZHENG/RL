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

def random_samples(prompt: list, fixed_ids: list, episodes, vocabulary,
                   print_interval, client, dataset,
                   format_prompt=". Format it strictly as entities separated by comma.", model_name="gpt-4o-mini", plot=True):
    data = utils.load_json_file(f"data/{dataset}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vocab_words = pd.read_csv(f"vocabulary/{vocabulary}.txt").columns.tolist()
    locs_all = list(range(len(prompt)))
    locs_avail = [loc for loc in locs_all if loc not in fixed_ids]
    PROMPTS, PREDICTIONS, SCORES, REWARDS = [], [], [], []
    
    # Initial prompt
    prompt_str = " ".join(prompt)
    prompt_complete = prompt_str + format_prompt
    scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
    SCORES.append(scores)
    REWARDS.append(reward)
    PREDICTIONS.append(predictions)
    PROMPTS.append(prompt)
    
    # Following prompts
    for episode in range(episodes):
        # Select a random location
        action_loc = random.choice(locs_avail)
        # Select a random word
        prompt[action_loc] = random.choice(vocab_words)
        # Updated prompt
        prompt_str = " ".join(prompt)
        prompt_complete = prompt_str + format_prompt
        # calculate reward
        scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt)
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

def do_training(prompt: list, fixed_ids: list, mean_random, std_random,
                episodes, d_embed, eps, random_action_start, 
                learning_rates, scheduler, scheduler_period, vocabulary, hiddens, seed, 
                print_interval, save_results, plot, client, dataset, 
                format_prompt=". Format it strictly as entities separated by comma.", model_name="gpt-4o-mini",
                smallest_lr=1e-6, decay_rate=0.5, plot_interval=50):
    #--------- Setup ---------------------
    utils.seed_everything(seed)                
    data = utils.load_json_file(f"data/{dataset}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vocab_words = pd.read_csv(f"vocabulary/{vocabulary}.txt").columns.tolist()
    print(f"Vocabulary size: {len(vocab_words)}")
    exp_name = f"d_{dataset}_p_{prompt}_ids_{fixed_ids}_stats_{mean_random}_{std_random}_v{vocabulary}_e_{episodes}_eps_{eps}_ras_{random_action_start}_l_{learning_rates}_sch_{scheduler}_schp_{scheduler_period}_h_{hiddens}_s_{seed}"
    
    #-------------- Initialization (the very first prompt designed by human) --------------------------
    prompt_str = " ".join(prompt)
    embedding = torch.tensor(utils.get_embedding(prompt_str, client, d_embed)).to(device)
    
    # Convert list of words to a sentence and add format prompt
    prompt_complete = prompt_str + format_prompt
    
    # ************************************** Training *******************************************************
    
    # ------------------Define networks -----------------------------------------------
    net_loc = rl.PolicyNet(input_dim=d_embed, hidden_dim=hiddens[0], output_dim=len(prompt)).to(device)
    optimizer_loc = optim.Adam(net_loc.parameters(), lr=learning_rates[0])
    net_word = rl.PolicyNet(input_dim=d_embed, hidden_dim=hiddens[1], output_dim=len(vocab_words)).to(device)
    optimizer_word = optim.Adam(net_word.parameters(), lr=learning_rates[1])
    if scheduler:
        lr_func = lambda itr: decay_rate ** itr
        scheduler = optim.lr_scheduler.LambdaLR(optimizer_word, lr_func)
    
    # ------------------- Calculate predictions and reward of the first prompt ----------------------------
    PROMPTS, PREDICTIONS, SCORES, REWARDS = [], [], [], []
    
    scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
    print(f"Initial prompt: {prompt}")
    print(f"Reward: {reward}\n")
    SCORES.append(scores)
    REWARDS.append(reward)
    PREDICTIONS.append(predictions)
    PROMPTS.append(prompt)
    
    # --------------------------- Training loops ---------------------------------
    for episode in range(episodes):
        lr_current = optimizer_word.param_groups[0]['lr']
        # Location
        action_loc_probs = net_loc.forward(embedding)
        for id in fixed_ids:
            action_loc_probs[id] = 1e-10
        action_loc_probs = action_loc_probs / torch.sum(action_loc_probs)
        dist_loc = torch.distributions.Categorical(action_loc_probs)
        action_loc = dist_loc.sample()
        log_prob_loc = dist_loc.log_prob(action_loc)
        # Word
        action_word_probs = net_word.forward(embedding)
        dist_word = torch.distributions.Categorical(action_word_probs)
        p = np.random.random()
        if episode > random_action_start and p < eps:
            print("Random action")
            action_word = torch.tensor(random.randrange(len(vocab_words))).to(device)
        else:
            action_word = dist_word.sample()
        log_prob_word = dist_word.log_prob(action_word)
        
        prompt[action_loc.item()] = vocab_words[action_word]

        prompt_str = " ".join(prompt)
        prompt_complete = prompt_str + format_prompt
        
        # calculate reward
        scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
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
        
        if episode % scheduler_period == 0 and episode != 0 and lr_current > smallest_lr:
            scheduler.step()
            print(f"Current learning rate: {lr_current}")
    
        SCORES.append(scores)
        REWARDS.append(reward)
        PREDICTIONS.append(predictions)
        PROMPTS.append(prompt)
    
        # New state
        embedding = torch.tensor(utils.get_embedding(prompt, client, d_embed)).to(device)
        
        if plot:
            if (episode + 1) % plot_interval == 0:
                plt.plot(REWARDS)
                plt.axhline(y=mean_random, color='r', linestyle='-')
                plt.fill_between(x=[0, episodes], y1=mean_random-std_random, y2=mean_random+std_random, color='r', alpha=.35)
                plt.xlim([0, episodes])
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
    
# def do_test(prompt, save_results, client, dataset,
#             format_prompt="Format it strictly as entities separated by comma.", model_name="gpt-4o-mini"):
#     data = utils.load_json_file(f"data/{dataset}.json")
#     prompt_complete = prompt + format_prompt
#     scores, predictions, reward = rl.reward_function(prompt_complete, data, client)
#     print(f"Tested prompt: {prompt}")
#     print(f"Predictions: {predictions}")
#     print(f"Scores: {scores}")
#     print(f"Reward: {reward}\n")
#     if save_results:
#         exp_name = f"test_{dataset}_p_{prompt}"
#         header = ['Prompt', 'Predictions', 'Scores', 'Reward']
#         with open(f'results/{exp_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(header)  # Write the header row
#             writer.writerow([prompt, ','.join(map(str, predictions)), scores, reward])
#     return None
    
# def do_training_exp_rply(prompt_length, replay_name, batch_size, mean_random, std_random, 
#                          episodes, learning_rates, vocabulary, word_len_min, hiddens, seed,
#                          print_interval, save_model=True, plot_loss=True, model_name="gpt-4o-mini"):
#     #--------- Setup ---------------------
#     utils.seed_everything(seed)                
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     encoding = tiktoken.encoding_for_model(model_name)
#     vocab = pd.read_csv("vocabulary.txt").columns.tolist()
#     print(f"Vocabulary size: {len(vocab)}")
#     model_name = f"h_{hiddens}_rply_{replay_name}_e_{episodes}"
    
#     # ------------------Define networks -----------------------------------------------
#     net_loc = rl.PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[0], output_dim=prompt_length).to(device)
#     optimizer_loc = optim.Adam(net_loc.parameters(), lr=learning_rates[0])
#     net_word = rl.PolicyNet(input_dim=prompt_length, hidden_dim=hiddens[1], output_dim=len(vocab)).to(device)
#     optimizer_word = optim.Adam(net_word.parameters(), lr=learning_rates[1])
    
#     #--------- Load replay buffer ---------------------
#     with open(f"buffer/{replay_name}.pkl", 'rb') as f:
#         replay_buffer = pickle.load(f)
#     print(f"Replay buffer {replay_name}.pkl loaded.")
    
#     dataset = rl.ReplayBufferDataset(replay_buffer)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     # ************************************** Training *******************************************************
    
#     losses_loc, losses_word = [], []
    
#     # --------------------------- Training loops ---------------------------------
#     for episode in range(episodes):
#         for batch in dataloader:
#             state, action_loc, log_prob_loc, action_word, log_prob_word, reward = batch
#             reward_n = (reward - mean_random) / std_random
#             loss_loc = torch.mean(-reward_n * log_prob_loc)
#             losses_loc.append(loss_loc.detach())
#             optimizer_loc.zero_grad()
#             loss_loc.backward()
#             optimizer_loc.step()
#             loss_word = torch.mean(-reward_n * log_prob_word)
#             losses_word.append(loss_word.detach())
#             optimizer_word.zero_grad()
#             loss_word.backward()
#             optimizer_word.step()
#         if episode % print_interval == 0:
#             print(f"Episode {episode + 1}/{episodes}, loss_loc: {loss_loc.item():.4f}, loss_word: {loss_word.item():.4f}")
            
#     if plot_loss:
#         plt.plot(losses_loc)
#         plt.xlabel("Episode")
#         plt.ylabel("losses_loc")
#         plt.title("losses_loc")
#         plt.show()
        
#         plt.plot(losses_word)
#         plt.xlabel("Episode")
#         plt.ylabel("losses_word")
#         plt.title("losses_word")
#         plt.show()
    
#     if save_model:
#         model_path = f'saved_models/{model_name}'
#         os.makedirs(model_path, exist_ok=True)
#         torch.save(net_loc.state_dict(), f'{model_path}/net_loc.pth')
#         torch.save(net_word.state_dict(), f'{model_path}/net_word.pth') 
