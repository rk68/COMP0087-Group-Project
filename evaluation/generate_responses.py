from datasets import load_dataset
import ollama
import re
import json
from tqdm import tqdm
import sqlite3
import pandas as pd
import argparse


'''
USAGE: python generate_responses.py dataset_name --subset dataset_subset dataset_split model_name num_examples local_db_name

e.g. python generate_responses.py aqua_rat --subset raw test gemma 10 gemma_test.db

note:
- do 'ollama pull model_name' before running
- ensure to check what dataset name / split on huggingface first


- currently only working for aqua_rat
- todo: add gsm8k support
'''


parser = argparse.ArgumentParser(description='Process some inputs')
 
parser.add_argument('dataset', type=str, help='Huggingface dataset name')
parser.add_argument('--subset', type=str, default=None, help='Subset of the huggingface dataset to use (optional)')
parser.add_argument('split', type=str, help='Split of the huggingface dataset to use e.g., train or test')
parser.add_argument('model_name', type=str, help='LLM Model Name')
parser.add_argument('num_examples', type=int, help='Number of data points to test')
parser.add_argument('db_name', type=str, help='Name of the SQLite database file')

args = parser.parse_args()

def deserialize_options(options_json):
    """Function to deserialize JSON strings back into Python lists"""
    return json.loads(options_json)

def create_table(dataset_name, conn):
    """SQL statement to create the table if it doesn't exist"""

    if dataset_name == 'aqua_rat': 
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS LLM_results (
            question TEXT,
            options TEXT,
            correct TEXT, 
            LLMs_rationale TEXT,
            LLMs_answer TEXT
        );

        """

    elif dataset_name == 'gsm8k':
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS LLM_results (
            question TEXT,
            answer TEXT, 
            LLMs_rationale TEXT,
            LLMs_answer TEXT
        );

        """

    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print("Table created successfully.")
    except Exception as e:
        print("Error creating table:", e)

def load_and_prepare_data(dataset_name, dataset_subset, dataset_split):
    """Load the dataset and prepare it for processing."""

    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    df = pd.DataFrame(dataset)

    if dataset_name == 'aqua_rat':
        # Serialize 'options' column to a string and drop 'rationale' if exists
        df['options'] = df['options'].apply(json.dumps)
        if 'rationale' in df.columns:
            df = df.drop(columns=['rationale'])
        # Initialize LLMs_rationale and LLMs_answer columns
        df['LLMs_rationale'] = pd.NA
        df['LLMs_answer'] = pd.NA
        return df
    
    elif dataset_name == 'gsm8k':
        df['LLMs_rationale'] = pd.NA
        df['LLMs_answer'] = pd.NA

        return df

def generate_responses(dataset_name, df, model_name, num_examples, conn):
    """Generate responses from the LLM and store them in the database."""
    tqdm.pandas()
    sampled_df = df.sample(n=num_examples, random_state=42)

    if dataset_name == 'aqua_rat':

        for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
            question = row['question']
            options_json = row['options']
            options = deserialize_options(options_json)
            options_text = ', '.join([f"{chr(65+j)}: {option}" for j, option in enumerate(options)])
            prompt = f"{question}\n\nOptions: {options_text}"
            response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
            sampled_df.at[index, 'LLMs_rationale'] = response['message']['content']

        sampled_df.to_sql('LLM_results', conn, if_exists='append', index=False)

    elif dataset_name == 'gsm8k':
            
        for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
            additional_prompt = "Answer the following question. At the end of your response, clearly state your answer in the format 'The answer is [value]' where the value is the numeric answer.\n"
            question = row['question']
            prompt = f"{additional_prompt + question}"
            response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
            sampled_df.at[index, 'LLMs_rationale'] = response['message']['content']
            
        sampled_df.to_sql('LLM_results', conn, if_exists='append', index=False)
        

def main():
    dataset_name = args.dataset
    dataset_subset = args.subset
    dataset_split = args.split
    model_name = args.model_name
    num_examples = args.num_examples
    db_name = args.db_name

    conn = sqlite3.connect(db_name)
    create_table(dataset_name, conn)

    df = load_and_prepare_data(dataset_name, dataset_subset, dataset_split)
    generate_responses(dataset_name, df, model_name, num_examples, conn)

    conn.close()

if __name__ == "__main__":
    main()
