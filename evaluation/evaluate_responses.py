import json
import re
import evaluate  # Import the evaluate library

# Load the exact match metric
exact_match = evaluate.load("exact_match")

# Load the responses from the JSON file
with open('responses.json', 'r') as f:
    data = json.load(f)

# Initialize variables to store predictions and correct answers
predictions = []
correct_answers = []

# Enhanced regex pattern to capture various phrasings of the correct answer
regex_pattern = r"(?:The correct answer is|The answer is|answer is|answer could be)\s*:? ?\[?([A-E])\]?"

for item in data:
    # Use the enhanced regex to extract the letter of the correct answer from the response
    match = re.search(regex_pattern, item["response"], re.IGNORECASE)

    if match:
        llm_answer = match.group(1).upper()
    else:
        llm_answer = "N/A"

    predictions.append(llm_answer)
    correct_answers.append(item["correct"].upper())

    # Update the item to indicate whether the prediction was correct
    item["predicted"] = llm_answer
    item["is_correct"] = llm_answer == item["correct"].upper()

# Calculate the exact match score
results = exact_match.compute(predictions=predictions, references=correct_answers)
em_score = results["exact_match"]

# Save the evaluation results (without the EM score)
with open('evaluation_results.json', 'w') as f:
    json.dump(data, f, indent=4)

print(f"Evaluation completed and saved. Overall Exact Match score: {em_score}%")
