#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from Super_Contrastive_Loss import SupConLoss
from utils import get_device


# In[2]:


with open('cls_emb.pkl', 'rb') as f:
    cls = pickle.load(f)
with open('feature_vectors.pkl', 'rb')as f:
    feature_vectors= pickle.load(f)


# In[ ]:


cls[0].size()


# In[4]:


response_df = pd.read_csv('final_data.csv')
map_dict = {'llama3.1-70b':0, 'mistral':1, 'gpt-4o-2024-05-13':2}
response_df['model_nums'] = response_df['model'].map(map_dict)


# In[5]:


embeddings = [torch.cat((cls[i].float(), torch.from_numpy(feature_vectors[i]).unsqueeze(0).float()), dim=1) for i in range(len(cls))]


# In[6]:


def extract_and_split(response_df, embeddings, temperature):
    temp_idx = response_df[response_df['temperature'] == temperature].index
    temp_embs = [embeddings[idx] for idx in temp_idx]
    temp_targs = [response_df['model_nums'][idx] for idx in temp_idx]
    
    return train_test_split(temp_embs, temp_targs, test_size=0.1, random_state=42)
    
temp_0_train, temp_0_test, temp_0_targs_train, temp_0_targs_test = extract_and_split(response_df, embeddings, 0)
temp_7_train, temp_7_test, temp_7_targs_train, temp_7_targs_test = extract_and_split(response_df, embeddings, 0.7)
temp_14_train, temp_14_test, temp_14_targs_train, temp_14_targs_test = extract_and_split(response_df, embeddings, 1.4)
temp_all_train, temp_all_test, temp_all_targs_train, temp_all_targs_test = train_test_split(embeddings, response_df['model_nums'], 
                                                                                            test_size=0.1, random_state=42)


# In[7]:


class FAM(nn.Module):
    def __init__(self, embed_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(embed_size, hidden_size)
        
    def init_weights(self):
        initrange = 0.2
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text):
        batch,  dim = text.size()
        feat = self.fc(torch.tanh(self.dropout(text.view(batch, dim))))
        feat = F.normalize(feat, dim=1)
        return feat


# In[8]:


class Projection(nn.Module):
    def __init__(self, hidden_size, projection_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, projection_size)
        self.ln = nn.LayerNorm(projection_size)
        self.bn = nn.BatchNorm1d(projection_size)
        self.init_weights()
    def init_weights(self):
        initrange = 0.01
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text):
        batch,  dim = text.size()
        return self.ln(self.fc(torch.tanh(text.view(batch, dim))))


# In[10]:


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_class, hidden_dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, feature):
        return self.fc(torch.tanh(feature))


# In[11]:


class WordEmbeddingDataset(Dataset):
    def __init__(self, cls_embs, targs):
        self.cls_embs = cls_embs
        self.targs = targs 

    def __len__(self):
        return len(self.cls_embs)

    def __getitem__(self, idx):
        return self.cls_embs[idx], self.targs[idx]


# In[12]:


BATCH_SIZE = 100
dataset_0 = WordEmbeddingDataset(temp_0_train, temp_0_targs_train)
dataset_0_test = WordEmbeddingDataset(temp_0_test, temp_0_targs_test)

dataset_7 =  WordEmbeddingDataset(temp_7_train, temp_7_targs_train)
dataset_7_test = WordEmbeddingDataset(temp_7_test, temp_7_targs_test)

dataset_14 = WordEmbeddingDataset(temp_14_train, temp_14_targs_train)
dataset_14_test = WordEmbeddingDataset(temp_14_test, temp_14_targs_test)

dataset_all = WordEmbeddingDataset(temp_all_train, temp_all_targs_train)
dataset_all_test = WordEmbeddingDataset(temp_all_test, temp_all_targs_test)

data_loader_0 = DataLoader(dataset_0, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_0_test = DataLoader(dataset_0_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_7 = DataLoader(dataset_7, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_7_test = DataLoader(dataset_7_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_14 = DataLoader(dataset_14, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_14_test = DataLoader(dataset_14_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_all = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_all_test = DataLoader(dataset_all_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# In[50]:


def train(fa_module, proj_module, supconloss_module, classifier, data_loader, optimizer, classifier_loss_fn):
    fa_module.train()
    proj_module.train()
    supconloss_module.train()
    classifier.train()

    batch_acc_cumulative = 0
    n_batches = 0
    train_loss = 0

    for _, data in tqdm(enumerate(data_loader)):
        n_batches += 1
        optimizer.zero_grad()

        cls_embs = data[0].squeeze(1)  # Assuming BERT CLS embeddings
        targets = data[1]

        # Forward pass through feature extractor and projection modules
        fam_output = fa_module(cls_embs)   
        proj_output = proj_module(fam_output)

        # Calculate the SupConLoss1 (replace SupConLoss here)
        supcon_loss = supconloss_module(proj_output, targets)

        # Forward pass through the classifier
        classifier_output = classifier(fam_output)  
        classifier_loss = classifier_loss_fn(classifier_output, targets)

        # Combine the SupCon loss and the classifier loss
        loss = supcon_loss + classifier_loss 

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update the cumulative loss and accuracy
        train_loss += loss.item()
        batch_predictions = classifier_output.argmax(1)
        batch_acc = (batch_predictions == targets).sum().item() / len(targets)
        batch_acc_cumulative += batch_acc

    average_acc = batch_acc_cumulative / n_batches
    average_loss = train_loss / n_batches

    return average_loss, average_acc

# In[51]:


def evaluate(fa_module, classifier, data_loader):
    fa_module.eval()  
    classifier.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in data_loader:
            cls_embs = data[0].squeeze(1)  
            targets = data[1].tolist()
            
            fam_output = fa_module(cls_embs)
            final_output = classifier(fam_output)
            preds = final_output.argmax(1).tolist()
            
            total += len(preds) 
            correct += np.sum(np.array(preds) == np.array(targets))  

    accuracy = correct / total if total > 0 else 0
    return accuracy


# In[ ]:


fam_0 = FAM(797, 256, 0.3)
proj_0 = Projection(256, 128)
supcon_0 = SupConLoss()
classifier_0 = Classifier(256, 3, 0.3)

optimizer = torch.optim.Adam(list(fam_0.parameters()) + 
                             list(proj_0.parameters()) + 
                             list(classifier_0.parameters()), lr=0.001)
classifier_loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
i = 0
best_acc = 0
while i <10:
    loss, acc = train(fam_0, proj_0, supcon_0, classifier_0, data_loader_0, optimizer, classifier_loss)  
    if acc > best_acc:
        best_acc = acc
        best_fam_0 = fam_0.state_dict()  
        best_proj_0 = proj_0.state_dict() 
        best_classifier = classifier_0.state_dict()
        i = 0
    else:
        i += 1
    scheduler.step()
    
fam_0.load_state_dict(best_fam_0)
proj_0.load_state_dict(best_proj_0)
classifier_0.load_state_dict(best_classifier)
test_accuracy = evaluate(fam_0, classifier_0, data_loader_0_test)
print('Test Set Accuracy: ' + str(test_accuracy*100) + '%')


# In[ ]:


fam_7 = FAM(797, 256, 0.3)
proj_7 = Projection(256, 128)
supcon_7 = SupConLoss()
classifier_7 = Classifier(256, 3, 0.3)

optimizer = torch.optim.Adam(list(fam_7.parameters()) + 
                             list(proj_7.parameters()) + 
                             list(classifier_7.parameters()), lr=0.001)
classifier_loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
i = 0
best_acc = 0
while i <10:
    loss, acc = train(fam_7, proj_7, supcon_7, classifier_7, data_loader_7, optimizer, classifier_loss)  
    if acc > best_acc:
        best_acc = acc
        best_fam_7 = fam_7.state_dict()  
        best_proj_7 = proj_7.state_dict() 
        best_classifier_7 = classifier_7.state_dict()
        i = 0
    else:
        i += 1
    scheduler.step()
    
fam_7.load_state_dict(best_fam_7)
proj_7.load_state_dict(best_proj_7)
classifier_7.load_state_dict(best_classifier_7)
test_accuracy = evaluate(fam_7, classifier_7, data_loader_7_test)
print('Test Set Accuracy: ' + str(test_accuracy*100) + '%')


# In[ ]:


fam_14 = FAM(797, 256, 0.3)
proj_14 = Projection(256, 128)
supcon_14 = SupConLoss()
classifier_14 = Classifier(256, 3, 0.3)

optimizer = torch.optim.Adam(list(fam_0.parameters()) + 
                             list(proj_0.parameters()) + 
                             list(classifier_0.parameters()), lr=0.001)
classifier_loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
i = 0
best_acc = 0
while i <10:
    loss, acc = train(fam_14, proj_14, supcon_14, classifier_14, data_loader_14, optimizer, classifier_loss)  
    if acc > best_acc:
        best_acc = acc
        best_fam_14 = fam_14.state_dict()  
        best_proj_14 = proj_14.state_dict() 
        best_classifier_14 = classifier_14.state_dict()
        i = 0
    else:
        i += 1
    scheduler.step()
    
fam_14.load_state_dict(best_fam_14)
proj_14.load_state_dict(best_proj_14)
classifier_14.load_state_dict(best_classifier)
test_accuracy = evaluate(fam_14, classifier_14, data_loader_14_test)
print('Test Set Accuracy: ' + str(test_accuracy*100) + '%')


# In[ ]:


# This cell is for training on all temperatures
fam_all = FAM(768, 256, 0.3)
proj_all = Projection(256, 128)
supcon_all = SupConLoss()
classifier_all = Classifier(256, 3, 0.3)
optimizer = torch.optim.Adam(list(fam_all.parameters()) + 
                             list(proj_all.parameters()) + 
                             list(supcon_all.parameters()) + 
                             list(classifier_all.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
for epoch in range(1, 20):
    loss, acc = train(fam_all, proj_all, supcon_all, classifier_all, data_loader_all)  
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
    scheduler.step()
test_accuracy = evaluate(fam_all, proj_all, supcon_all, classifier_all, data_loader_all_test)


# In[ ]:




