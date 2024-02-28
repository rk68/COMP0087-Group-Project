from datasets import load_dataset
import ollama
import re
import json
from tqdm import tqdm
import sqlite3
import pandas as pd
import argparse


'''
USAGE: python generate_responses.py dataset_name --subset dataset_subset dataset_split model_name num_examples

e.g. python generate_responses.py aqua_rat --subset raw test gemma 10

currently only working for aqua_rat. 
'''


parser = argparse.ArgumentParser(description='Process some inputs')
 
parser.add_argument('dataset', type=str, help='Huggingface dataset name')
parser.add_argument('--subset', type=str, default=None, help='Subset of the huggingface dataset to use (optional)')
parser.add_argument('split', type=str, help='Split of the huggingface dataset to use e.g., train or test')
parser.add_argument('model_name', type=str, help='LLM Model Name')
parser.add_argument('num_examples', type=int, help='Number of data points to test')

args = parser.parse_args()

def deserialize_options(options_json):
    """Function to deserialize JSON strings back into Python lists"""
    return json.loads(options_json)

def create_table(conn):
    """SQL statement to create the table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS LLM_results (
        question TEXT,
        options TEXT,
        correct TEXT, 
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
    # Serialize 'options' column to a string and drop 'rationale' if exists
    df['options'] = df['options'].apply(json.dumps)
    if 'rationale' in df.columns:
        df = df.drop(columns=['rationale'])
    # Initialize LLMs_rationale and LLMs_answer columns
    df['LLMs_rationale'] = pd.NA
    df['LLMs_answer'] = pd.NA
    return df

def generate_responses(df, model_name, num_examples, conn):
    """Generate responses from the LLM and store them in the database."""
    tqdm.pandas()
    sampled_df = df.sample(n=num_examples, random_state=42)

    for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
        question = row['question']
        options_json = row['options']
        options = deserialize_options(options_json)
        options_text = ', '.join([f"{chr(65+j)}: {option}" for j, option in enumerate(options)])
        prompt = f"{question}\n\nOptions: {options_text}"
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        sampled_df.at[index, 'LLMs_rationale'] = response['message']['content']
        # 'LLMs_answer' remains NA here; will be filled after evaluation

    # No need to drop 'correct' column, as it's now intended to be part of the database
    sampled_df.to_sql('LLM_results', conn, if_exists='append', index=False)

def main():
    dataset_name = args.dataset
    dataset_subset = args.subset
    dataset_split = args.split
    model_name = args.model_name
    num_examples = args.num_examples

    conn = sqlite3.connect('llm_responses.db')
    create_table(conn)

    df = load_and_prepare_data(dataset_name, dataset_subset, dataset_split)
    generate_responses(df, model_name, num_examples, conn)

    conn.close()

if __name__ == "__main__":
    main()
