# Import dependencies
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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
dev_indices, val_indices = train_test_split(indices, test_size = 0.1, random_state = RANDOM_STATE)

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
        return ([embeddings[idx] for idx in val_indices if response_df['temperature'][idx]==temp],
            [model_nums[idx] for idx in val_indices if response_df['temperature'][idx]==temp])

temp_0_train, temp_0_test, temp_0_targs_train, temp_0_targs_test = extract_and_split_dev(0)
temp_7_train, temp_7_test, temp_7_targs_train, temp_7_targs_test = extract_and_split_dev(0.7)
temp_14_train, temp_14_test, temp_14_targs_train, temp_14_targs_test = extract_and_split_dev(1.4)
temp_0_val, temp_0_val_targs = extract_and_split_dev(0, False)
temp_7_val, temp_7_val_targs = extract_and_split_dev(0.7, False)
temp_14_val, temp_14_val_targs = extract_and_split_dev(1.4, False)
temp_all_val, temp_all_val_targs = [embeddings[idx] for idx in val_indices],[model_nums[idx] for idx in val_indices]
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
        return ([embeddings[idx] for idx in val_indices if response_df['temperature'][idx]==temp],
            [model_nums[idx] for idx in val_indices if response_df['temperature'][idx]==temp])

temp_0_train, temp_0_test, temp_0_targs_train, temp_0_targs_test = extract_and_split_dev(0)
temp_7_train, temp_7_test, temp_7_targs_train, temp_7_targs_test = extract_and_split_dev(0.7)
temp_14_train, temp_14_test, temp_14_targs_train, temp_14_targs_test = extract_and_split_dev(1.4)
temp_0_val, temp_0_val_targs = extract_and_split_dev(0, False)
temp_7_val, temp_7_val_targs = extract_and_split_dev(0.7, False)
temp_14_val, temp_14_val_targs = extract_and_split_dev(1.4, False)
temp_all_val, temp_all_val_targs = [embeddings[idx] for idx in val_indices],[model_nums[idx] for idx in val_indices]

temp_all_embs = [embeddings[idx] for idx in dev_indices]
temp_all_targs = [model_nums[idx] for idx in dev_indices]
temp_all_train, temp_all_test, temp_all_targs_train, temp_all_targs_test = train_test_split(
    temp_all_embs, temp_all_targs, test_size = 0.2, random_state = RANDOM_STATE)


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
        batch, dim = text.size()
        feat = self.fc(torch.tanh(self.dropout(text.view(batch, dim))))
        feat = F.normalize(feat, dim=1)
        return feat

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

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

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


class WordEmbeddingDataset(Dataset):
    """
    Custom dataset object to feed into the dataloader
    """

    def __init__(self, embs, targs):
        """
        Class initializer

        embs: concatenated CLS embeddings and feature vectors
        targs: base truth model numbers
        """
        self.embs = embs
        self.targs = targs

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        """
        Method used by data loader to output data

        idx: random batch index from dataloader

        Returns:
        Tuple with batch embeddings at [0] and batch base truth targets at [1]
        """
        return self.embs[idx], self.targs[idx]

# Use batch size of 100
BATCH_SIZE = 100

# Create datasets to be used in data loaders
dataset_0 = WordEmbeddingDataset(temp_0_train, temp_0_targs_train)
dataset_0_test = WordEmbeddingDataset(temp_0_test, temp_0_targs_test)

dataset_7 =  WordEmbeddingDataset(temp_7_train, temp_7_targs_train)
dataset_7_test = WordEmbeddingDataset(temp_7_test, temp_7_targs_test)

dataset_14 = WordEmbeddingDataset(temp_14_train, temp_14_targs_train)
dataset_14_test = WordEmbeddingDataset(temp_14_test, temp_14_targs_test)

dataset_all = WordEmbeddingDataset(temp_all_train, temp_all_targs_train)
dataset_all_test = WordEmbeddingDataset(temp_all_test, temp_all_targs_test)

# Create all dataloaders
data_loader_0_train = DataLoader(dataset_0, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_0_test = DataLoader(dataset_0_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_7_train = DataLoader(dataset_7, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_7_test = DataLoader(dataset_7_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_14_train = DataLoader(dataset_14, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_14_test = DataLoader(dataset_14_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

data_loader_all_train = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_all_test = DataLoader(dataset_all_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class ModelGenerator():
    """
    Because there are dropout layers in the networks, each training session will have inherent stochasticity. We
    will build many models and pick the best one
    """

    def __init__(self, data_loader_train, data_loader_test):
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test

    def train(self):
        """
        Model training function

        fa_module: FAM class object used for initial feature extraction from initial CLS and Feature tensors
        proj_module: Projection class object used to reduce FAM features for contrastive learning loss function
        supconloss_module: SupConLoss class object that calculates contrastive loss
        classifer: Classifier class object that outputs predictions from FAM features
        data_loader: DataLoader class object that splits data into batches for model training
        optimizer: optimizer object which helps models with parameters converge to proper feature weights
        classifier_loss_fn: loss function which helps the classifer converge to proper feature weights

        Returns:
            Tuple of average loss at [0] and accuracy at [1] for each epoch
        """
        # Set all networks to train mode
        self.fam.train()
        self.proj.train()
        self.supcon_loss.train()
        self.classifier.train()

        # Declare variables to measure batch performance
        correct = 0
        total_targets = 0
        n_batches = 0
        train_loss = 0

        for data in self.data_loader_train:
            # Start model training
            n_batches += 1
            self.optimizer.zero_grad()
            embs = data[0].squeeze(1)
            targets = data[1]
            """
            Model flow: FAM -> projection head -> supcon loss. Predictions are made from FAM features, while
            Supconloss is calculated from the projection head.
            """
            fam_output = self.fam(embs)
            proj_output = self.proj(fam_output)
            supcon_loss = self.supcon_loss(proj_output, targets)
            classifier_output = self.classifier(fam_output)
            classifier_loss = self.classifier_loss(classifier_output, targets)

            # Use a combined supconloss and classifer loss
            loss = supcon_loss + classifier_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy for the batch
            preds = classifier_output.argmax(1)
            correct += np.sum(np.array(preds) == np.array(targets))
            total_targets += len(targets)

        # Calculate overall accuracy and loss
        accuracy = correct / total_targets
        average_loss = train_loss / n_batches

        return average_loss, accuracy

    def evaluate(self):
        """
        fa_module: optimized FAM object used to generate features
        classifier: optimized Classifier object used to classify features from fa_module
        data_loader: DataLoader class object that splits data into batches for model training

        Returns:
            accuracy of evaluation
        """
        # Set networks to eval mode
        self.fam.eval()
        self.classifier.eval()

        correct = 0
        total = 0

        # Evaluation the FAM and Classifier
        with torch.no_grad():
            for data in self.data_loader_test:
                embs = data[0].squeeze(1)
                targets = data[1].tolist()

                fam_output = self.fam(embs)
                final_output = self.classifier(fam_output)
                preds = final_output.argmax(1).tolist()

                total += len(preds)
                correct += np.sum(np.array(preds) == np.array(targets))

        accuracy = correct / total
        return accuracy

    def gen_model(self, hidden_size_2, supconloss_temp):
        """
        Generates the best model based on test set accuracy
        """
        # Declare constants for networks
        HIDDEN_SIZE_1 = 256
        DROPOUT_PERCENT = 0.3
        LEARNING_RATE = 0.001
        STEP_SIZE = 20
        GAMMA = 0.5
        self.hidden_size_2 = hidden_size_2
        self.supconloss_temp = supconloss_temp
        NUM_CLASSES = 3
        STOP_EARLY_NUM = 10
        NUM_MODELS = 10

        # Keep track of the best test accuracy
        best_test_acc = 0
        for i in tqdm(range(NUM_MODELS)):
            self.fam = FAM(NUM_FEATURES, HIDDEN_SIZE_1, DROPOUT_PERCENT)
            self.proj = Projection(HIDDEN_SIZE_1, self.hidden_size_2)
            self.classifier = Classifier(HIDDEN_SIZE_1, NUM_CLASSES, DROPOUT_PERCENT)
            self.optimizer = optimizer = torch.optim.Adam(list(self.fam.parameters()) +
                                                          list(self.proj.parameters()) +
                                                          list(self.classifier.parameters()),
                                                          lr=LEARNING_RATE)
            self.scheduler = scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE,
                                                                         gamma=GAMMA)
            self.classifier_loss = nn.CrossEntropyLoss()
            self.supcon_loss = SupConLoss(self.supconloss_temp)
            i = 0
            best_acc = 0
            # Utilize 'stopping early' when 10 successive models have failed to improve
            while i < STOP_EARLY_NUM:
                loss, acc = self.train()
                if acc > best_acc:
                    best_acc = acc
                    best_fam = self.fam.state_dict()
                    best_proj = self.proj.state_dict()
                    best_classifier = self.classifier.state_dict()
                    i = 0
                else:
                    i += 1
                scheduler.step()

            # Load the best training model for test set evaluation
            self.fam.load_state_dict(best_fam)
            self.proj.load_state_dict(best_proj)
            self.classifier.load_state_dict(best_classifier)
            test_accuracy = self.evaluate()

            if test_accuracy > best_test_acc:
                # If this model performs better than previous models, update overall best
                best_test_acc = test_accuracy
                overall_best_fam = best_fam
                overall_best_proj = best_proj
                overall_best_classifier = best_classifier
        self.fam.load_state_dict(overall_best_fam)
        self.proj.load_state_dict(overall_best_proj)
        self.classifier.load_state_dict(overall_best_classifier)

    def test_accuracy(self, data_loader_test):
        self.fam.eval()
        self.classifier.eval()

        correct = 0
        total = 0

        # Evaluation the FAM and Classifier
        with torch.no_grad():
            for data in self.data_loader_test:
                embs = data[0].squeeze(1)
                targets = data[1].tolist()

                fam_output = self.fam(embs)
                final_output = self.classifier(fam_output)
                preds = final_output.argmax(1).tolist()

                total += len(preds)
                correct += np.sum(np.array(preds) == np.array(targets))

        accuracy = correct / total
        return accuracy



