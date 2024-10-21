
import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterSampler
from typing import List, Tuple, Union




def get_device():
    """
    Set the device to 'cuda' or 'mps' or 'cpu' depending on the availability of a GPU.
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


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



class MisinfoTsneAnalyzer:

    def __init__(self, raw_data_path: str, model_type, embeds_path: str, features_embeds_path: str, model_data_temp: Union[str, float, int], random_state=42):
        
        self.test_params = {
            'n_components': [2],
            'perplexity': [5, 30, 50],  # Emphasizing local and global structures
            'n_iter': [1000],  # Different levels of precision
            'learning_rate': [100, 200, 500, 1000],  # Varying the learning rate for convergence speed
            'init': ['pca', 'random'],  # Testing different initialization methods
            'metric': ['euclidean', 'cosine'],  # Exploring different distance metrics
            'early_exaggeration': [4, 8, 12],  # Impacting initial cluster separation
            'angle': [0.5],  # Trade-off between speed and accuracy
            'method': ['barnes_hut'],  # Methods suitable for different dataset sizes
            'random_state': [42]  # Fixed for reproducibility; can be adjusted if desired
        }

        self.results_destination = 'tsne_models_objects_and_results'
        self.model_type = model_type
        self.embeds_path = embeds_path
        self.features_embeds_path = features_embeds_path
        self.raw_data_path = raw_data_path
        self.model_data_temp = model_data_temp
        self.model = None
        self.extended_embeds_tensor = None
        self.model_labels = None
        self.temp_labels = None
        self.viz_df = None
        self.random_state = random_state
        self.tsne_params = None
        self.tsne = None

        self.model_path_template = 'models_and_objects/temp_{}_models.pth'
        state_dict_key_template = '{}_state_dict'
        device = get_device()

        # Determine the correct path based on temperature
        template_fill_val = self.model_data_temp
        temp_to_path_int = lambda temp_float: int(float(temp_float) * 10)
        try:
            template_fill_val = temp_to_path_int(self.model_data_temp)
        except:
            pass

        # Initialize the model based on the type
        if self.model_type == 'classifier':
            self.model = Classifier(256, 3, 0.3).to(device)
        elif self.model_type == 'fam':
            self.model = FAM(797, 256, 0.3).to(device)
        else:
            raise ValueError('model_type must be either "classifier" or "fam"')

        # Load the model state
        state_dict = torch.load(self.model_path_template.format(template_fill_val), map_location=device)
        classifier_state = state_dict[state_dict_key_template.format(self.model_type)]
        self.model.load_state_dict(classifier_state)

        # Load the embeddings
        with open(self.embeds_path, 'rb') as f:
            embeds = pickle.load(f)

        # Load the feature embeddings (relevant only for FAM)
        with open(self.features_embeds_path, 'rb') as f:
            feats_embeds = pickle.load(f)

        # Load the data
        data = pd.read_csv(self.raw_data_path)
        self.model_labels = data['model'].values
        self.temp_labels = data['temperature'].values

        # if the model data temp is a digit, we will find index of the temp in the raw data 'temperature' column.
        # Then we will use those index to retrieve/filter the embeddings, features embeddings, model labels and temp labels
        if str(model_data_temp).isdigit():
            model_data_temp = float(model_data_temp)  # Ensure it's the correct type for comparison
            temp_index = data[data['temperature'] == model_data_temp].index
            embeds = np.array(embeds)[temp_index]
            feats_embeds = np.array(feats_embeds)[temp_index]
            self.model_labels = self.model_labels[temp_index]
            self.temp_labels = self.temp_labels[temp_index]

        # Convert embeddings to tensors and concatenate if necessary
        embeds_array = np.array(embeds)
        feats_embeds_array = np.array(feats_embeds)
        embeds_tensor = torch.tensor(embeds_array, dtype=torch.float32).to(device)
        feats_embeds_tensor = torch.tensor(feats_embeds_array, dtype=torch.float32).to(device)

        # Flatten if embeddings are 3D with a second dimension of 1
        if len(embeds_tensor.size()) == 3 and embeds_tensor.size(1) == 1:
            embeds_tensor = embeds_tensor.view(embeds_tensor.size(0), -1)

        # Concatenate embeddings
        self.extended_embeds_tensor = torch.cat((embeds_tensor, feats_embeds_tensor), dim=1)
    
    @staticmethod
    def save_tsne_model(tsne, tsne_params, model_type, temp_value, output_dir="tsne_stuff", random_state=42):
        # remove random_state from tsne_params if it exists
        tsne_params.pop('random_state', None)
        # 
        filename_str = ', '.join([f"{key}: {value}" for key, value in tsne_params.items()])
        sanitized_filename_str = "".join(e for e in filename_str if e.isalnum() or e == "_")
        
        # Create the filename using the sanitized suptitle, temp value, and model_type
        filename = f"tsne_model_{sanitized_filename_str}_model_{model_type}_temp_{temp_value}.pkl"
        filepath = os.path.join(output_dir, filename)
        
        # Save the t-SNE model
        with open(filepath, 'wb') as f:
            pickle.dump(tsne, f)
    
    def run_tsne_with_params(self, tsne_params, save_model=False):
        # Ensure model is in evaluation mode and generate transformed embeddings
        with torch.no_grad():
            self.model.eval()
            if self.model_type == 'classifier':
                transformed_embeds = self.model(self.extended_embeds_tensor).cpu()
            elif self.model_type == 'fam':
                transformed_embeds = self.model(self.extended_embeds_tensor).cpu().numpy()

        # Perform t-SNE with the given parameters
        tsne = TSNE(**tsne_params)
        tsne_embeds = tsne.fit_transform(transformed_embeds)

        if save_model:
            self.save_tsne_model(tsne, tsne_params, self.model_type, self.model_data_temp)

        # Create a dataframe for visualization
        self.viz_df = pd.DataFrame(tsne_embeds, columns=['x', 'y'])
        self.viz_df['model'] = self.model_labels
        self.viz_df['temperature'] = self.temp_labels
        return self.viz_df, tsne
    
    def tsne_plot(self, viz_n: int, save_fig=True, show_fig=True):

        # remove random_state from tsne_params if it exists
        filename_tsne_params = self.tsne_params.copy()
        filename_tsne_params.pop('random_state', None)
        filename_str = ', '.join([f"{key}: {value}" for key, value in filename_tsne_params.items()])
        sanitized_filename_str = "".join(e for e in filename_str if e.isalnum() or e == "_")

        # Create the filename using the sanitized suptitle, temp value, and model_type
        filename = f"tsne_plot_{sanitized_filename_str}_model_{self.model_type}_temp_{self.model_data_temp}.png"
        filepath = os.path.join(self.results_destination, filename)

        # if we have more than 500 points, we will randomly sample 500 points
        if len(self.viz_df) > viz_n:
            self.viz_df = self.viz_df.sample(viz_n, random_state=self.random_state)

        # plot the t-SNE
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=self.viz_df, x='x', y='y', hue='model', style='temperature', palette='tab10')
        plt.suptitle('t-SNE of model embeddings')
        plt.title(filename_str)
        plt.xlabel('t-SNE component 1')
        plt.ylabel('t-SNE component 2')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_fig:
            plt.savefig(filepath)
        if show_fig:
            plt.show()
        plt.close()

    def search_tsne_param_space(self, silhouette_sample, n_iter, verbose=False):
        # Generate random combinations of t-SNE parameters
        param_combinations = list(ParameterSampler(self.test_params, n_iter=n_iter, random_state=self.random_state))

        self.best_score = -1
        self.tsne_params = None
        self.viz_df = None
        self.tsne = None

        # List to store information for each run
        results = []

        for i, params in enumerate(param_combinations):
            # Run t-SNE with the specified parameters
            viz_df, tsne = self.run_tsne_with_params(tsne_params=params, save_model=False)
            
            if verbose:
                print("t-SNE computation complete")

            silhouette_score_x = viz_df[['x', 'y']].values
            silhouette_labels = viz_df['model'].values

            # Calculate the silhouette score for the current t-SNE embedding
            score = silhouette_score(X=silhouette_score_x,
                                     labels=silhouette_labels,
                                     sample_size=silhouette_sample,
                                     random_state=self.random_state)

            # Store the results for this parameter set
            result_entry = {
                'model_type': self.model_type,
                'model_data_temp': self.model_data_temp,
                'silhouette_score': score,
            }
            # Add each t-SNE parameter as its own column
            for key, value in params.items():
                result_entry[f'tsne_param_{key}'] = value
            results.append(result_entry)

            if verbose:
                print(f"Run {i + 1}/{n_iter} - Silhouette score: {score}")

            # Save the best score and parameters
            if score > self.best_score:
                self.best_score = score
                self.tsne_params = params
                self.viz_df = viz_df
                self.tsne = tsne


                # Save the t-SNE model and plot as it's the best so far
                self.save_tsne_model(tsne, params, self.model_type, self.model_data_temp, output_dir=self.results_destination)
                self.tsne_plot(viz_n=500, save_fig=True, show_fig=False)

                if verbose:
                    print(f"New best silhouette score: {self.best_score}")

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.results_destination, f"tsne_search_results.csv")

        # Check if the CSV file already exists
        if os.path.isfile(results_csv_path):
            # If it exists, read the existing CSV and append the new results
            existing_df = pd.read_csv(results_csv_path)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)

            # remove redundant rows
            combined_df.drop_duplicates(inplace=True)
        else:
            # If it doesn't exist, the new results are the combined DataFrame
            combined_df = results_df

        # Save the combined DataFrame back to the CSV
        combined_df.to_csv(results_csv_path, index=False)

        if verbose:
            print(f"Best silhouette score: {self.best_score} with parameters:\n{self.tsne_params}\n") 
            print(f"Results saved to: {results_csv_path}")