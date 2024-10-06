import copy
import hashlib
import json
import logging
import os
import requests
import time
import pandas as pd
from datetime import datetime
from openai import OpenAI



def process_batch_output_file(file_path):
    """
    Function to process the batch output JSONL file and extract relevant features into a DataFrame.

    :param file_path: Path to the JSONL file.
    :return: A pandas DataFrame with extracted features.
    """
    rows = []

    # Open and read the file line by line
    with open(file_path, 'r') as f:
        for line in f:
            row = json.loads(line.strip())
            rows.append(row)

    # Prepare data for the dataframe
    data = []
    model_name = None

    for row in rows:
        custom_id = row.get('custom_id')
        response = row.get('response', {})
        body = response.get('body', {})
        model_name = body.get('model')  # Extract the model name
        content = body.get('choices', [{}])[0].get('message', {}).get('content', '')  # Extract the content
        usage = body.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)  # Extract the prompt tokens
        completion_tokens = usage.get('completion_tokens', 0)  # Extract the completion tokens

        # Add the extracted values to the data list
        data.append({
            'hash': custom_id,
            'content': content,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        })

    # Create a dataframe from the data
    df = pd.DataFrame(data)

    # Rename the 'content' column with the model name (if available)
    if model_name:
        df.rename(columns={'content': model_name}, inplace=True)

    return df


class OpenAIMisinfoBatchManager:
    def __init__(self, temp=None, top_p=None, api_key =None, template_choice = None, max_tokens = None, model="gpt-4o"):
        """
        A class to generate and manage misinfo generation batches using OpenAI's API 
        :param temp: The temperature parameter for the model
        :param api_key: The API key for the OpenAI API
        :param model: The model to use for generating misinformation
        """
        
        self.batch_id = None
        self.input_file_id = None
        self.output_file_id = None
        self.error_file_id = None
        self.requests_jsonl = None
        self.model = model
        self.temp = temp
        self.max_tokens = max_tokens
        self.template_choice = template_choice
        self.client = OpenAI(api_key=api_key)
        self.jsonl_path_template = "misinfo_requests_{}.jsonl"
        self.chat_request_template = {
            "custom_id": None,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": None}
                ]
            }
        }
        # if max_tokens is provided, add it to the request body
        if self.max_tokens:
            self.chat_request_template['body']['max_tokens'] = self.max_tokens
        logging.info("Initialized OpenAIMisinfoBatch Obj with model: %s, template: %s, temp: %s, max_tokens: %s", self.model, self.template_choice, self.temp, self.max_tokens)
    
    @staticmethod
    def _generate_timestamp():
        """Generate a timestamp for the current time"""
        return datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _generate_requests_jsonl(self, prompts_df: pd.DataFrame):
        """
        Create a JSONL file containing the requests to send to the OpenAI API
        :param prompts_df: A DataFrame containing the prompts to generate misinformation for. required columns: 'prompt_for_generation', 'hash'
        :return: The path to the JSONL file
        """
        logging.info("Generating JSONL requests from DataFrame. DataFrame shape: %s", prompts_df.shape)
        requests = []
        jsonl_path = self.jsonl_path_template.format(self._generate_timestamp())
        for index, row in prompts_df.iterrows():
            full_prompt = ''
            if pd.isnull(row['prompt_for_generation']):
                print(f"Warning: Missing prompt at index {index}")
                continue

            else:
                full_prompt = row['prompt_for_generation']

            request = copy.deepcopy(self.chat_request_template)
            request['custom_id'] = row['hash']
            request['body']['messages'][1]['content'] = full_prompt
            requests.append(request)

        with open(jsonl_path, 'w') as f:
            for request in requests:
                json.dump(request, f)
                f.write('\n')
        
        logging.info("JSONL requests file created at %s", jsonl_path)
        return jsonl_path
    
    def send_batch_misinfo_request(self, prompts_df: pd.DataFrame):
        """
        Send a batch request to OpenAI to generate misinformation
        :param prompts_df: A DataFrame containing the prompts to generate misinformation. Required columns: 'prompt_for_generation', 'hash'
        """
        try:    
            jsonl_path_file = self._generate_requests_jsonl(prompts_df)

            # send the file to the OpenAI API
            batch_input_file = self.client.files.create(
                file=open(jsonl_path_file, 'rb'),
                purpose='batch'
            )

            self.input_file_id = batch_input_file.id
            logging.info("Batch input file ID: %s", self.input_file_id)

            batch_result = self.client.batches.create(
                input_file_id=self.input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": f"misinfo generation for {self.template_choice} prompts. {self._generate_timestamp()}"
                }
            )
            self.batch_id = batch_result.id
            logging.info("Batch created with ID: %s", self.batch_id)
            return batch_result
        
        except Exception as e:
            logging.error("Error sending batch request: %s", e)
            raise e
    
    def retrieve_batch_results(self, output_filepath, error_filepath, max_wait_time=None, status_check_interval=10):
        """
        Retrieve the results of a batch request
        :param output_filepath: The path to save the output file
        :param error_filepath: The path to save the error file
        :param max_wait_time: The maximum time to wait for the batch to complete in seconds
        :param status_check_interval: The interval to check the status of the batch in seconds
        """
        
        try:
            if not self.batch_id:
                raise ValueError("Batch ID is required to retrieve the results of a batch")
            
            logging.info("Retrieving batch results for Batch ID: %s", self.batch_id)
            
            status = ''
            batch_ended_statuses = ['completed', 'failed', 'cancelled']
            start_time = time.time()

            while status not in batch_ended_statuses:
                try:
                    retrieved_batch = self.client.batches.retrieve(self.batch_id)
                    status = retrieved_batch.status
                    running_time = time.time() - start_time
                    logging.info("Batch status: %s, Time elapsed: %.2f minutes", status, running_time / 60)
                    if max_wait_time and running_time > max_wait_time:
                        logging.error("Max wait time exceeded. Exiting.")
                        raise TimeoutError("Max wait time exceeded.")
                    
                    if status in batch_ended_statuses:
                        break

                    time.sleep(status_check_interval)
                except requests.exceptions.ConnectionError as e:
                    # Log network error and continue the loop
                    logging.error("Network error while checking batch status: %s", str(e))
                    time.sleep(status_check_interval)

            if status == 'completed':
                logging.info("Batch completed successfully.")
                self.output_file_id = retrieved_batch.output_file_id
                self.error_file_id = retrieved_batch.error_file_id
                if self.output_file_id:
                    output_file_response = self.client.files.content(self.output_file_id)
                    with open(output_filepath, 'w') as f:
                        f.write(output_file_response.text)
                    logging.info("Output file saved to %s", output_filepath)
                else:
                    logging.info("No output file found for batch.")

                if self.error_file_id:
                    error_file_response = self.client.files.content(self.error_file_id)
                    with open(error_filepath, 'w') as f:
                        f.write(error_file_response.text)
                    logging.info("Error file saved to %s", error_filepath)
                else:
                    logging.info("No error file found for batch.")
            else:
                logging.error("Batch failed with status: %s", status)
                if retrieved_batch.errors:
                    for batch_error in retrieved_batch.errors.data:
                        logging.error("Error: %s", batch_error.message)
                raise RuntimeError("Batch failed.")

        except Exception as e:
            logging.error("Error while retrieving batch results: %s", str(e))
            raise
