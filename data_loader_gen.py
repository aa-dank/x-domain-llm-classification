import torch
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
from SCL import WordEmbeddingDataset

# Import pre-computed BERT CLS vectors and Feature Vectors
with open('cls_emb.pkl', 'rb') as f:
    cls = pickle.load(f)
with open('feature_vectors.pkl', 'rb')as f:
    feature_vectors= pickle.load(f)

# The model needs numerical classes as base truth targets. Convert model names to 0, 1, 2
response_df = pd.read_csv('final_data.csv')
map_dict = {'llama3.1-70b':0, 'mistral':1, 'gpt-4o-2024-05-13':2}
response_df['model_nums'] = response_df['model'].map(map_dict)
model_nums = response_df['model_nums']

# Concatenate the CLS and feature vectors
embeddings = [torch.cat((cls[i].float(), torch.from_numpy(feature_vectors[i]).unsqueeze(0).float()), dim=1) for i in range(len(cls))]

# Calculate the number of features to be used later as input to model
NUM_FEATURES = embeddings[0].size(1)

# Create a development and validation set by splitting indices
RANDOM_STATE = 42
indices = [i for i in range(len(embeddings))]
dev_indices, test_indices = train_test_split(indices, test_size = 0.1, random_state = RANDOM_STATE)

def extract_and_split_dev(temp, dev=True):
    """
    Returns split train, test, and dev sets based on temperature

    temp: float indicating temperature
    dev: boolean indicating if the return set is the dev set or validation
    """
    if dev:
        temp_embs = [embeddings[idx] for idx in dev_indices if response_df['temperature'][idx]==temp]
        temp_targs = [model_nums[idx] for idx in dev_indices if response_df['temperature'][idx]==temp]
        return train_test_split(temp_embs, temp_targs, test_size=0.2, random_state=RANDOM_STATE)
    if not dev:
        return ([embeddings[idx] for idx in test_indices if response_df['temperature'][idx]==temp],
            [model_nums[idx] for idx in test_indices if response_df['temperature'][idx]==temp])

temp_0_train, temp_0_val, temp_0_targs_train, temp_0_targs_val = extract_and_split_dev(0)
temp_7_train, temp_7_val, temp_7_targs_train, temp_7_targs_val = extract_and_split_dev(0.7)
temp_14_train, temp_14_val, temp_14_targs_train, temp_14_targs_val = extract_and_split_dev(1.4)
temp_0_test, temp_0_targs_test = extract_and_split_dev(0, False)
temp_7_test, temp_7_targs_test = extract_and_split_dev(0.7, False)
temp_14_test, temp_14_targs_test = extract_and_split_dev(1.4, False)
temp_all_test, temp_all_targs_test = [embeddings[idx] for idx in test_indices],[model_nums[idx] for idx in test_indices]

temp_all_embs = [embeddings[idx] for idx in dev_indices]
temp_all_targs = [model_nums[idx] for idx in dev_indices]
temp_all_train, temp_all_val, temp_all_targs_train, temp_all_targs_val = train_test_split(
    temp_all_embs, temp_all_targs, test_size = 0.2, random_state = RANDOM_STATE)

with open('temp_0_test.pkl', 'wb') as f:
    pickle.dump(temp_0_test, f)
with open('temp_0_targs_test.pkl', 'wb') as f:
    pickle.dump(temp_0_targs_test, f)

with open('temp_7_test.pkl', 'wb') as f:
    pickle.dump(temp_7_test, f)
with open('temp_7_targs_test.pkl', 'wb') as f:
    pickle.dump(temp_7_targs_test, f)

with open('temp_14_test.pkl', 'wb') as f:
    pickle.dump(temp_14_test, f)
with open('temp_14_targs_test.pkl', 'wb') as f:
    pickle.dump(temp_14_targs_test, f)

with open('temp_all_test.pkl', 'wb') as f:
    pickle.dump(temp_all_test, f)
with open('temp_all_targs_test.pkl', 'wb') as f:
    pickle.dump(temp_all_targs_test, f)

# Use batch size of 100
BATCH_SIZE = 100

# Create datasets to be used in data loaders
dataset_0 = WordEmbeddingDataset(temp_0_train, temp_0_targs_train)
dataset_0_val = WordEmbeddingDataset(temp_0_val, temp_0_targs_val)

dataset_7 =  WordEmbeddingDataset(temp_7_train, temp_7_targs_train)
dataset_7_val = WordEmbeddingDataset(temp_7_val, temp_7_targs_val)

dataset_14 = WordEmbeddingDataset(temp_14_train, temp_14_targs_train)
dataset_14_val = WordEmbeddingDataset(temp_14_val, temp_14_targs_val)

dataset_all = WordEmbeddingDataset(temp_all_train, temp_all_targs_train)
dataset_all_val = WordEmbeddingDataset(temp_all_val, temp_all_targs_val)

dataset_0_7 = WordEmbeddingDataset(temp_0_train+temp_7_train, temp_0_targs_train+temp_7_targs_train)
dataset_0_14 = WordEmbeddingDataset(temp_0_train+temp_14_train, temp_0_targs_train+temp_14_targs_train)
dataset_7_14 = WordEmbeddingDataset(temp_7_train+temp_14_train, temp_7_targs_train+temp_14_targs_train)

dataset_0_test = WordEmbeddingDataset(temp_0_test, temp_0_targs_test)
dataset_7_test = WordEmbeddingDataset(temp_7_test, temp_7_targs_test)
dataset_14_test = WordEmbeddingDataset(temp_14_test, temp_14_targs_test)
dataset_all_test = WordEmbeddingDataset(temp_all_test, temp_all_targs_test)
# Create all dataloaders
data_loader_0_train = DataLoader(dataset_0, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_0_val = DataLoader(dataset_0_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_7_train = DataLoader(dataset_7, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_7_val = DataLoader(dataset_7_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_14_train = DataLoader(dataset_14, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_14_val = DataLoader(dataset_14_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_all_train = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_all_val = DataLoader(dataset_all_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_0_7 = DataLoader(dataset_0_7, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_0_14 = DataLoader(dataset_0_14, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_7_14 = DataLoader(dataset_7_14, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_0_test = DataLoader(dataset_0_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_7_test = DataLoader(dataset_7_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_14_test = DataLoader(dataset_14_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_all_test = DataLoader(dataset_all_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)