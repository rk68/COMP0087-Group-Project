import sqlite3
import pandas as pd
import argparse
import re
import evaluate


'''
USAGE: python evaluate_responses.py database_path --eval_type eval_method

eval_method is either numeric (for llms that output a numeric answer) or letter (for letter multiple choice questions)
gi
e.g. python evaluate_responses.py llm_responses.db --eval_type letter   

note: Ensure that llm_responses.db has been created by using generate_responses.py first.
'''

def eval_numeric(text):
    """Extract a numeric answer from the LLM response."""
    processed_text = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", processed_text)
    if numbers:
        return str(numbers[-1].replace(',', ''))
    return None  # Return None if no numeric answer is found

def eval_letter(text):
    """Extract a letter (A-E) answer from the LLM response."""
    split_result = text.split("answer is", 1)
    if len(split_result) == 2:
        _, after_text = split_result
        match = re.search(r"([A-E])", after_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None  # Return None if no letter answer is found

def update_llm_answers(conn, df, eval_type):
    """Update the LLMs_answer column based on the eval_type."""
    for index, row in df.iterrows():
        llm_rationale = row['LLMs_rationale']
        if eval_type == 'numeric':
            df.at[index, 'LLMs_answer'] = eval_numeric(llm_rationale)
        elif eval_type == 'letter':
            df.at[index, 'LLMs_answer'] = eval_letter(llm_rationale)
    df.to_sql('LLM_results', conn, if_exists='replace', index=False)

def calculate_exact_match(conn, eval_type):
    """Calculate the exact match score and print the final score."""
    df = pd.read_sql_query("SELECT correct, LLMs_answer FROM LLM_results", conn)
    exact_match = evaluate.load("exact_match")

    # Ensure predictions and references are lists of strings
    predictions = [str(answer) if answer is not None else "" for answer in df['LLMs_answer'].tolist()]
    references = [str(correct) if correct is not None else "" for correct in df['correct'].tolist()]

    em_score = exact_match.compute(predictions=predictions, references=references, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact Match Score ({eval_type} evaluation): {em_score * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Evaluate and update LLM responses stored in an SQL database.')
    parser.add_argument('database', type=str, help='Database name where the responses are stored')
    parser.add_argument('--eval_type', type=str, choices=['numeric', 'letter'], required=True, help='Evaluation type: numeric or letter')
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    df = pd.read_sql_query("SELECT * FROM LLM_results", conn)
    update_llm_answers(conn, df, args.eval_type)
    calculate_exact_match(conn, args.eval_type)
    conn.close()

if __name__ == "__main__":
    main()
