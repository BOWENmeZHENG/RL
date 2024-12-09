import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import glob

# Calculation
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def preprocess_string(string, remove_stopwords=True, lemmatize=True):
    """
    Preprocess a string: lowercase, remove punctuation, remove stopwords, and lemmatize.
    
    Args:
        string (str): The input string.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to apply lemmatization.
    
    Returns:
        set: A set of processed tokens.
    """
    # Convert to lowercase
    string = string.lower()
    # Remove punctuation
    string = re.sub(r'[^\w\s]', '', string)
    # Tokenize
    tokens = word_tokenize(string)
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return set(tokens)

def calculate_iou(string1, string2):
    """
    Calculate IoU (Intersection over Union) for two preprocessed strings.
    
    Args:
        string1 (str): First string.
        string2 (str): Second string.
    
    Returns:
        float: IoU score.
    """
    tokens1 = preprocess_string(string1)
    tokens2 = preprocess_string(string2)
    
    # Compute intersection and union
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    # Calculate IoU
    iou = len(intersection) / len(union) if union else 0.0
    return iou

def calculate_precision_recall_with_iou(predicted, ground_truth, iou_threshold=0.5):
    """
    Calculate precision and recall for predicted and ground truth lists using IoU > threshold.
    
    Args:
        predicted (list): List of predicted strings.
        ground_truth (list): List of ground truth strings.
        iou_threshold (float): IoU threshold for considering a prediction correct.
    
    Returns:
        tuple: Precision and Recall as floats.
    """
    # Track matches
    matched_ground_truth = set()
    correct_predictions = 0
    
    # Evaluate predictions
    for pred in predicted:
        for gt in ground_truth:
            if gt not in matched_ground_truth:  # Avoid double-matching ground truth entries
                iou = calculate_iou(pred, gt)
                if iou > iou_threshold:
                    correct_predictions += 1
                    matched_ground_truth.add(gt)  # Mark this ground truth as matched
                    break  # Stop checking other ground truth for this prediction
    
    # Calculate precision and recall
    precision = correct_predictions / len(predicted) if predicted else 0.0
    recall = correct_predictions / len(ground_truth) if ground_truth else 0.0
    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return precision, recall, f1_score


# Data    
def add_entry_to_json(filename, new_paragraph, new_materials, new_properties):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {'paragraph': [], 'entities': []}

    data['paragraph'].append(new_paragraph)
    data['entities'].append({'MATERIAL': new_materials, 'PROPERTY': new_properties})

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_json_file(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{filename}'.")
        return None

def clear_json_file(filename):
    # Check if the file exists
    if os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump({'paragraph': [], 'entities': []}, f, indent=4)
        print(f"Content of '{filename}' cleared successfully.")
    else:
        print(f"Error: File '{filename}' not found.")

# LLM
def extract_named_entities(prompt, text, client, temperature=0.0, max_tokens=100):
    """
    Extract named entities from text using a language model client.

    Args:
        prompt (str): Instruction for the model to identify specific named entities.
        text (str): The input text containing the content to analyze.
        client (object): The model client for generating the response.
        temperature (float, optional): Sampling temperature, controls randomness. Default is 0.0.
        max_tokens (int, optional): Maximum number of tokens to generate. Default is 100.

    Returns:
        str: The model's response containing the extracted entities.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# Plotting
def extract_filenames_with_prefix(directory, prefix):
    matching_files = glob.glob(os.path.join(directory, prefix + '*'))
    filenames = [os.path.basename(file) for file in matching_files]
    return filenames

def load_results(exp: str):
    exp_names = extract_filenames_with_prefix("results", exp)
    all_results = []
    # Load results in dataframes
    for exp_name in exp_names:
        result = pd.read_csv(f"results/{exp_name}")["Reward"]
        all_results.append(result)
    return all_results

def plot_results(all_results, fontsize=12, figsize=(10, 6)):
    """Plots the mean and standard deviation of test results."""

    # Calculate mean and standard deviation
    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    # Plotting
    plt.figure(figsize=figsize)
    x = np.arange(len(mean_results))
    plt.plot(x, mean_results)
    plt.fill_between(x, mean_results - std_results, mean_results + std_results, alpha=0.2)

    plt.xlabel("Episode", fontsize=fontsize)
    plt.ylabel("Mean F1 score", fontsize=fontsize)
    plt.title("Rewards over Episodes", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()