from datasets import load_dataset
import ollama
import re
import evaluate  # Import the evaluate library

# specify dataset
dataset_name = "aqua_rat"

# specify teacher model
model_name = "mistral"
# Load the dataset
dataset = load_dataset(dataset_name, split='test')

# Load the exact match metric
exact_match = evaluate.load("exact_match")

# Process the first 5 prompts
responses = []
predictions = []  # Store predictions for exact match calculation
correct_answers = []  # Store correct answers for exact match calculation

for i in range(min(5, len(dataset))):  # Ensure not to exceed dataset length
    # Extracting the prompt and options
    prompt = dataset[i]['question']
    options = dataset[i]['options']
    correct = dataset[i]['correct']  # Correct answer
    options_text = ', '.join([f"{chr(65+j)}:{option}" for j, option in enumerate(options)])  # Formatting options with letters
    
    # Construct the prompt for OLLaMa
    enhanced_prompt = f"Question: {prompt}\nAnswer choices: {options_text}.\nThink step by step. You must indicate the correct answer by stating 'The correct answer is [LETTER]'. You must use that format."

    # Generate a response using OLLaMa
    ollama_response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': enhanced_prompt}])
    response_text = ollama_response['message']['content']

    # Extract the letter of the correct answer using a regex that matches the specific phrase pattern
    regex_pattern = r"The correct answer is ([A-E])"
    match = re.search(regex_pattern, response_text, re.IGNORECASE)

    if match:
        llm_answer = match.group(1).upper()
    else:
        llm_answer = "N/A"  # In case no match is found or the pattern is not followed

    predictions.append(llm_answer)
    correct_answers.append(correct.upper())

    # The correctness check will be done using exact match metric later
    responses.append((enhanced_prompt, response_text, correct))

# Calculate exact match score
results = exact_match.compute(predictions=predictions, references=correct_answers)
em_score = results["exact_match"]
print(f"Exact match score: {em_score}%")

# Print the responses and whether they were correct according to exact match
for enhanced_prompt, response, correct in responses:
    is_correct = "Correct" if predictions[responses.index((enhanced_prompt, response, correct))].upper() == correct.upper() else "Incorrect"
    print(f"Prompt: {enhanced_prompt}\nOLLAMa's Answer: {response}\nCorrectness: {is_correct} (Correct Answer was: {correct})\n")
