from datasets import load_dataset
import ollama
import re
import evaluate  # Import the evaluate library
from tqdm import tqdm
import sqlite3
import pandas as pd
import json



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

    # Execute the SQL statement
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print("Table created successfully.")
    except Exception as e:
        print("Error creating table:", e)




def get_metrics(conn):
    """This function will allow the user to input their dataset ano other metrics and then create a 
    randomly sampled training set of questions to ask the LLM"""

    # specify dataset
    dataset_name = str(input('Dataset: '))

    # Load the dataset and convert training split into pandas df
    dataset = load_dataset(dataset_name)
    df_train = pd.DataFrame(dataset.get('train'))

    # Shuffle the entire df_train DataFrame
    df_randomised = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Serialize the 'options' column to a string
    df_randomised['options'] = df_randomised['options'].apply(json.dumps)

    # Drop the 'rationale' column 
    df_randomised = df_randomised.drop(columns=['rationale'])
    df_randomised['LLMs_rationale'] = pd.NA
    df_randomised['LLMs_answer'] = pd.NA

    df_randomised.to_sql('LLM_results', conn, if_exists='replace', index=False)


    return df_randomised




def eval_llm(df_dataset, conn):

    # specify teacher model
    model_name = str(input('Teacher LLM: '))

    # specify how many points to be tested
    number_points = int(input('Number of data points to sample: '))

    # specify prompt to be used for LLM
    user_prompt = str(input('Prompt: '))

    # Load the exact match metric
    exact_match = evaluate.load("exact_match")      

    # Count the number of rows where the values in the 'correct' column match the values in the 'LLMs answer' column
    ### THIS NEXT LINE WILL BE USED WHEN GETTING A FINAL DATASET
    #num_valid_rows_count = len(df_dataset[df_dataset['correct'] == df_dataset['LLMs answer']])
    num_valid_rows = df_dataset['LLMs_answer'].notna() & (df_dataset['LLMs_answer'] != "N/A")
    num_valid_rows_count = num_valid_rows.sum()

    # Initialize tqdm with the total number of iterations
    progress_bar = tqdm(total=number_points)

    if num_valid_rows_count != 0:
        progress_bar.update(num_valid_rows_count)

    while num_valid_rows_count <= number_points:
        # Find the index of the next row where 'LLMs answer' is None
        next_none_index = df_dataset[df_dataset['LLMs_answer'].isnull()].index.min()

        prompt = df_dataset.loc[next_none_index, 'question']
        # options = df_dataset.loc[next_none_index, 'options']
        # options_text = ', '.join([f"{chr(65+j)}:{option}" for j, option in enumerate(options)])  # Formatting options with letters
        options_json = df_dataset.loc[next_none_index, 'options']
        options = deserialize_options(options_json)

        # Formatting options with letters
        options_text = ', '.join([f"{chr(65+j)}:{option}" for j, option in enumerate(options)])

        # Construct the prompt for OLLaMa
        enhanced_prompt = user_prompt.replace("{prompt}", prompt)
        enhanced_prompt = enhanced_prompt.replace("{options_text}", options_text) 

        # Generate a response using OLLaMa
        ollama_response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': enhanced_prompt}])
        response_text = ollama_response['message']['content']

        # Before entering the conditional logic, initialize llm_answer with a default value
        llm_answer = "N/A"  # Default value indicating no answer or unrecognized format

        # Then your existing logic to attempt to extract and set a more specific value
        split_result = response_text.split("answer is", 1)
        if len(split_result) == 2:
            before_text, after_text = split_result
            regex_pattern = r"([A-E])"
            match = re.search(regex_pattern, after_text, re.IGNORECASE)
            if match:
                llm_answer = match.group(1).upper()
                # There's no need to repeat the assignment of llm_answer to itself after this point

        # add LLMs MCQ answer and rationale to the dataframe
        df_dataset.loc[next_none_index, 'LLMs_answer'] = llm_answer
        df_dataset.loc[next_none_index, 'LLMs_rationale'] = response_text

        # Count the number of rows in the 'LLMs answer' column that are not None or "N/A"
        num_valid_rows = df_dataset['LLMs_answer'].notna() & (df_dataset['LLMs_answer'] != "N/A")
        num_valid_rows_new = num_valid_rows.sum()

        ### THIS NEXT LINE WILL BE USED WHEN GETTING A FINAL DATASET
        #num_valid_rows_count = len(df_dataset[df_dataset['correct'] == df_dataset['LLMs answer']])

        if num_valid_rows_new != num_valid_rows_count :
            progress_bar.update(1)
            num_valid_rows_count  = num_valid_rows_new

    df_dataset.to_sql('LLM_results', conn, if_exists='replace', index=False)


    # Close the progress bar
    progress_bar.close()








def main():
    # specify database name
    database_name = str(input('Database name: '))

    # Create a SQLite database connection
    conn = sqlite3.connect(database_name)

    # Create the table if it doesn't exist
    create_table(conn)

    # Check if there is existing data in the database
    df_dataset = pd.read_sql_query("SELECT * FROM LLM_results", conn)
    if df_dataset.empty:
        # Initialize an empty DataFrame if there is no existing data
        df_dataset = get_metrics(conn)
    eval_llm(df_dataset, conn)

    # Close the database connection when done
    conn.close()

    



if __name__ == "__main__":
    main()



