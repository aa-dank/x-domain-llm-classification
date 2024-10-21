# Import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
import torch.utils.data as data
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

class Model():
    """
    Because there are dropout layers in the networks, each training session will have inherent stochasticity. We
    will build many models and pick the best one
    """

    def __init__(self, data_loader_train, data_loader_val, filepath=None,
                 hidden_size=None):
        RANDOM_STATE = 42
        DROPOUT_PERCENT = 0.3
        NUM_CLASSES = 3
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.num_features = data_loader_train.dataset[0][0].size(1)
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.preds = []
        self.targs = []



        if filepath:
            model_data = torch.load(filepath)
            self.fam = FAM(self.num_features, hidden_size, DROPOUT_PERCENT)
            self.classifier = Classifier(hidden_size, NUM_CLASSES, DROPOUT_PERCENT)
            self.fam.load_state_dict(model_data['fam_state_dict'])
            self.classifier.load_state_dict(model_data['classifier_state_dict'])

    def train(self, data_loader):
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

        for data in data_loader:
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

    def evaluate(self, data_loader):
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
        accuracies = []
        # Evaluation the FAM and Classifier
        with torch.no_grad():
            for data in data_loader:
                embs = data[0].squeeze(1)
                targets = data[1].tolist()

                fam_output = self.fam(embs)
                final_output = self.classifier(fam_output)
                preds = final_output.argmax(1).tolist()
                self.preds.append(preds)
                self.targs.append(targets)
                total += len(preds)
                correct += np.sum(np.array(preds) == np.array(targets))

        accuracy = correct / total
        return accuracy

    def gen_model(self, hidden_size, supcontemp):
        """
        Generates the best model based on test set accuracy
        """
        # Declare constants for networks
        self.hidden_size = hidden_size
        self.supcontemp = supcontemp
        HIDDEN_SIZE_2 = 128
        DROPOUT_PERCENT = 0.3
        NUM_CLASSES = 3
        LEARNING_RATE = 0.001
        STEP_SIZE = 20
        GAMMA = 0.5
        STOP_EARLY_NUM = 10
        NUM_MODELS = 10
        k = 5
        best_val_acc = 0
        kfold = KFold(n_splits=k, shuffle=True)
        dataset = self.data_loader_train.dataset

        for i in tqdm(range(NUM_MODELS)):
            fold_accs = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                train_subsampler = data.SubsetRandomSampler(train_idx)
                val_subsampler = data.SubsetRandomSampler(val_idx)

                train_loader = torch.utils.data.DataLoader(dataset, sampler=train_subsampler,
                                                           batch_size=self.data_loader_train.batch_size)
                val_loader = torch.utils.data.DataLoader(dataset, sampler=val_subsampler,
                                                         batch_size=self.data_loader_train.batch_size)

                self.fam = FAM(self.num_features, self.hidden_size, DROPOUT_PERCENT)
                self.proj = Projection(self.hidden_size, HIDDEN_SIZE_2)
                self.classifier = Classifier(self.hidden_size, NUM_CLASSES, DROPOUT_PERCENT)
                self.optimizer = optimizer = torch.optim.Adam(list(self.fam.parameters()) +
                                                              list(self.proj.parameters()) +
                                                              list(self.classifier.parameters()),
                                                              lr=LEARNING_RATE)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=GAMMA)
                self.classifier_loss = nn.CrossEntropyLoss()
                self.supcon_loss = SupConLoss(supcontemp)

                # Early stopping
                i = 0
                best_training_acc = 0
                while i < STOP_EARLY_NUM:
                    loss, acc = self.train(train_loader)
                    if acc > best_training_acc:
                        best_training_acc = acc
                        best_fam = self.fam.state_dict()
                        best_proj = self.proj.state_dict()
                        best_classifier = self.classifier.state_dict()
                        i = 0
                    else:
                        i += 1
                    self.scheduler.step()

                # Evaluate on the validation set
                self.fam.load_state_dict(best_fam)
                self.proj.load_state_dict(best_proj)
                self.classifier.load_state_dict(best_classifier)
                val_accuracy = self.evaluate(val_loader)
                fold_accs.append(val_accuracy)

            if np.mean(fold_accs) > best_val_acc:
                best_val_acc = np.mean(fold_accs)
                self.overall_best_fam = best_fam
                self.overall_best_proj = best_proj
                self.overall_best_classifier = best_classifier
                print('**MODEL UPDATED**')
                print('Average Training Accuracy: ' + str(best_training_acc))
                print('Best Validation Accuracy: ' + str(np.mean(fold_accs)))
        self.fam.load_state_dict(self.overall_best_fam)
        self.proj.load_state_dict(self.overall_best_proj)
        self.classifier.load_state_dict(self.overall_best_classifier)

    def get_bootstrap_dataloader(self, embs, targs):
        BATCH_SIZE = 100
        np.random.seed(42)
        num_samples = len(embs)
        bootstrap_indices = self.rng.choice(np.arange(num_samples), size=num_samples, replace=True)
        sample_embs = [embs[idx] for idx in bootstrap_indices]
        sample_targs = [targs[idx] for idx in bootstrap_indices]
        dataset = WordEmbeddingDataset(sample_embs, sample_targs)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        return data_loader

    def run_bootstrap(self, num_iterations, embs, targs):
        accuracies = []
        for _ in tqdm(range(num_iterations)):
            data_loader = self.get_bootstrap_dataloader(embs, targs)
            accuracy = self.evaluate(data_loader)
            accuracies.append(accuracy)
        return accuracies

    def save_model(self, filepath):
        # Save the best models to disk
        torch.save({'fam_state_dict': self.overall_best_fam,
                    'proj_state_dict': self.overall_best_proj,
                    'classifier_state_dict': self.overall_best_classifier,
                    }, filepath)

    def reset_preds_targs(self):
        self.preds = []
        self.targs = []