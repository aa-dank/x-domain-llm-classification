# LLM Attribution: Challenges and Insights Across Model Stochasticity

## Introduction
Large Language Models (LLMs) have ushered in a new era in artificial intelligence, enabling sophisticated natural language processing tasks and generating human-like text with unprecedented fluency. However, as the proliferation of AI-generated content increases, so does the challenge of attributing text to specific models—a critical concern for authorship verification, intellectual property rights, and mitigating misinformation.

Detailed in our accompanying paper, LLM Attribution, we investigate various approaches to discern textual characteristics that may indicate a specific LLM's signature. Through careful experimentation and analysis, we seek to understand the nuances between different models and the potential for reliable attribution.

This repository provides the complete codebase used in our research, including data processing scripts, model implementations, and evaluation tools. Our work seeks to shed light on the complexities of LLM attribution and to offer a foundation for further research in this critical area.

## File Guide


| Category                | Description                                                         | Files                                                                                                 |
|-------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Dataset Generation**  | ChatGPT data generation                                             | [ChatGPT_Data_Generation.py](ChatGPT_Data_Generation.py) <br> [ChatGPT_Data_Generation.ipynb](ChatGPT_Data_Generation.ipynb) |
| **Dataset Generation**  | LLAMA data generation                                               | [llama_data_gen.ipynb](llama_data_gen.ipynb)                                                          |
| **Dataset Generation**  | MISTRAL data generation                                             | [mistral_data_generation.ipynb](mistral_data_generation.ipynb)                                        |
| **Data Preprocessing**  | Generate numerical features from text with embeddings               | [generate_word_embeddings.ipynb](generate_word_embeddings.ipynb)                                      |
| **Data Preprocessing**  | Generate numerical features by extracting stylometric features      | [stylometry_vector_gen.ipynb](stylometry_vector_gen.ipynb)                                            |
| **Supervised Learning** | Logistic Regression classification approach                         | [Logistic_regression_final.ipynb](Logistic_regression_final.ipynb)                                    |
| **Supervised Learning** | Random Forest classification approach                               | [Clean_Random_Forest_model.ipynb](Clean_Random_Forest_model.ipynb)                                    |
| **Unsupervised Learning** | t-SNE analysis tools and code                                     | [misinfo_tsne.py](misinfo_tsne.py) <br> [misinfo_tsne.ipynb](misinfo_tsne.ipynb)                      |
| **Supervised Learning** | Supervised Contrastive Learning (SCL) model generation              | [SCL.py](SCL.py)                                                                                      |
| **Supervised Learning** | Data loader generation and data splitting                           | [data_loader_gen.py](data_loader_gen.py)                                                              |
| **Supervised Learning** | Analysis of SCL models                                              | [SCL_model_analysis.ipynb](SCL_model_analysis.ipynb)                                                  |


## Data Generation

### ChatGPT Data Generation

To generate data using the ChatGPT model, use the [ChatGPT_Data_Generation.ipynb](ChatGPT_Data_Generation.ipynb) notebook. This notebook handles batching of prompts and manages interactions with the OpenAI API.

**Steps:**

1. **Setup:**
   - Install required dependencies:
     ```bash
     pip install openai pandas requests
     ```
   - Set your OpenAI API key in a secure location, such as a `creds.py` file:
     ```python
     OPENAI_KEY = 'your-api-key'
     ```

2. **Prepare Prompts:**
   - Ensure `prompts.csv` is located in the `data/` directory. This CSV file should contain the following columns:
     - `prompt_for_generation`: The actual text prompt.
     - `hash`: A unique identifier for each prompt.
     - `type`: The type of prompt (e.g., `"rewrite"`, `"paraphrase"`, `"open_ended"`).

3. **Configure Parameters:**
   - Adjust settings in the configuration section of the script or notebook:
     ```python
     prompts_csv_path = "data/prompts.csv"
     domain = "paraphrase"  # Choose from "rewrite", "paraphrase", or "open_ended"
     model_temp = 0.7       # Temperature settings: 0.0, 0.7, 1.4
     batch_size = 50        # Number of prompts per batch
     ```

4. **Run the Script or Notebook:**
   - Execute the Python script directly:
     ```bash
     python ChatGPT_Data_Generation.py
     ```
   - Alternatively, open `ChatGPT_Data_Generation.ipynb` in Jupyter and run all cells sequentially.

5. **Batch Processing Details:**
   - The prompts are processed in batches, with each batch sent asynchronously to the API. This helps manage rate limits and reduces the risk of timeouts.
   - The script handles retries for failed requests, ensuring that all prompts are processed even if temporary network issues occur.

6. **Retrieve and Process Results:**
   - The generated responses are saved in JSONL format and then converted to a CSV file for analysis. The output files include:
     - `chatgpt_responses.jsonl`: Raw API responses.
     - `chatgpt_dataset.csv`: Processed data ready for analysis, including columns for `hash`, `temperature`, `prompt_type`, and `response_text`.

7. **Repeat for Different Settings:**
   - To generate data across various conditions, adjust the `domain` and `model_temp` parameters and rerun the script or notebook.
   - This iterative approach allows for comprehensive data collection across different LLM configurations.


**Notes:**

- Batches have their own, additional token limit which needs to be monitored.

### LLAMA Data Generation

To generate data using the LLAMA model, use the [llama_data_gen.ipynb](llama_data_gen.ipynb) notebook. This notebook handles batching of prompts and manages interactions with the LLAMA API.

**Steps:**


1. **Setup Dependencies:**
   - Install required packages:
     ```bash
     pip install llamaapi pandas
     ```
   - Configure your API key securely:
     ```python
     api_key = 'your-api-key'
     ```

2. **Prepare Input Data:**
   - Place `prompts.csv` in the `data/` directory with the following columns:
     - `prompt_for_generation`: The input prompt text.
     - `hash`: A unique identifier for each prompt.
     - `type`: One of `"rewrite"`, `"paraphrase"`, or `"open_ended"`.

3. **Configure Parameters:**
   - Set parameters in the notebook’s configuration cell:
     ```python
     prompts_csv_path = "data/prompts.csv"
     model = 'llama3.1-70b'
     temps = [0.0, 0.7, 1.4]  # Temperature settings
     top_p = 0.9              # Top-p sampling parameter
     batch_size = 50          # Number of prompts per batch
     ```

4. **Run the Notebook:**
   - Open `llama_data_gen.ipynb` and execute all cells sequentially.
   - The notebook processes prompts in batches, iterating through each temperature setting and sending asynchronous requests to the LLAMA API.
   - The batching mechanism includes retry logic for any failed API calls, ensuring robust data generation.

5. **Output Files:**
   - The responses are collected in a DataFrame and saved to `llama_data.csv` after each generation batch. The output file includes:
     - `hash`: Unique identifier for the prompt.
     - `temp`: Temperature setting used.
     - `model`: Model name (e.g., `llama3.1-70b`).
     - `response_text`: Generated text or error message.
     - `datetime`: Timestamp of the generation.

6. **Repeat for Different Configurations:**
   - To explore different models or parameters, adjust the `model` and `temps` variables and rerun the notebook.
   - This iterative process allows for comprehensive data collection under various conditions, aiding in diverse analysis scenarios.

**Notes:**
- Monitor API rate limits when using large batch sizes to avoid throttling.
- Logs are generated during execution to track progress and provide insights for troubleshooting any issues encountered during data generation.

### Mistral Data Generation

To generate data using the Mistral model, use the [mistral_data_generation.ipynb](mistral_data_generation.ipynb) notebook. This notebook processes prompts and manages interactions with the Mistral AI API.

**Steps:**

1. **Setup Dependencies:**
   - Install the required packages:
     ```bash
     pip install mistralai pandas
     ```
   - Set your Mistral AI API key securely:
     ```python
     api_key = 'your-api-key'
     ```

2. **Prepare Input Data:**
   - Ensure `prompts.csv` is located in the `data/` directory. It should contain the following columns:
     - `prompt_for_generation`: The input prompt text.
     - `hash`: A unique identifier for each prompt.
     - `type`: One of `"rewrite"`, `"paraphrase"`, or `"open_ended"`.

3. **Configure Parameters:**
   - Edit the configuration in the notebook:
     ```python
     api_key = 'your-api-key'        # Mistral API key
     model = 'mistral-large-2407'    # Model version
     temperatures = [0.0, 0.7, 1.4]  # Temperature settings
     batch_size = 50                 # Number of prompts per batch
     dataset_name = "mistral_dataset_raw.csv"
     ```

4. **Run the Notebook:**
   - Open `mistral_data_generation.ipynb` and execute all cells sequentially.
   - The notebook processes prompts in batches, iterating over each temperature setting. It sends asynchronous API requests, handling retries automatically for failed calls.

5. **Retrieve and Process Results:**
   - The responses are stored in `mistral_dataset_raw.csv` after each generation batch. The processed output is saved to `mistral_ai_dataset.csv`, containing:
     - `hash`: Unique identifier for each prompt.
     - `temperature`: Temperature used for generation.
     - `model`: Name of the model (e.g., `mistral-large-2407`).
     - `response_text`: The generated text response.
     - `datetime`: Timestamp of the generation.
   - Post-processing steps include removing duplicate entries and cleaning the data:
     ```python
     clean_df = raw_df.drop_duplicates(subset=['temperature', 'hash'])
     clean_df.to_csv("mistral_ai_dataset.csv", index=False)
     ```

6. **Repeat for Different Settings:**
   - To generate data using different models or temperature configurations, update the `model` and `temperatures` variables in the notebook and rerun it.
   - This allows you to collect diverse datasets under various conditions for comprehensive analysis.

**Additional Notes:**
- Be mindful of API rate limits, especially when using high batch sizes.
- Logs are created during execution to monitor the status of each batch and to help with troubleshooting any issues.


## Summary of Results and Research Findings

1. **Impact of Model Stochasticity**:
   - The temperature setting, which controls the randomness of LLM output, significantly affects the accuracy of authorship attribution. Higher temperatures (e.g., 1.4) introduce more variability in the generated text, making it harder to correctly identify the source model.
   - Our analysis showed a decline in classifier performance as the temperature increased, indicating the difficulty of distinguishing LLMs based on more creative outputs.

2. **Supervised Contrastive Learning (SCL) Evaluation**:
   - Despite using advanced Supervised Contrastive Learning techniques, the results did not show a marked improvement over traditional classifiers like Random Forest and Logistic Regression.
   - The performance gap suggests that simpler models may be competitive when the input features include strong stylometric and embedding-based characteristics.

3. **Unsupervised Learning and Clustering**:
   - We applied dimensionality reduction techniques (e.g., t-SNE) to explore the separability of LLM-generated texts. However, the embeddings did not exhibit meaningful clustering across different LLMs, especially as model sophistication increased.
   - This lack of clear clustering suggests that advanced LLMs may converge in their stylistic outputs, complicating attribution tasks.

4. **Dataset and Feature Engineering Insights**:
   - The dataset used included responses generated at different temperature settings across multiple LLMs. Short responses and incoherent outputs at high temperatures posed challenges for training and evaluation.
   - We combined CLS embeddings from BERT with stylometric features to enhance the input representation. Ablation studies revealed that stylometric features alone did not significantly boost performance, indicating potential redundancy when combined with strong embedding features.

5. **Comparison with Previous Work**:
   - Unlike prior studies, our findings did not demonstrate clear separability between LLMs using embedding-based approaches. This may be attributed to the use of more advanced, state-of-the-art models in our dataset, which appear to exhibit convergent behavior.
   - The results indicate that distinguishing between more sophisticated LLMs is increasingly challenging, highlighting the need for novel methods that go beyond traditional embedding and stylometric analysis.

6. **Ethical Considerations**:
   - While our methods aim to aid in detecting and attributing AI-generated text, they could also be misused to evade detection or obscure the source of generated content. Researchers and practitioners should exercise caution when deploying these models, ensuring that proper verification steps are taken.
