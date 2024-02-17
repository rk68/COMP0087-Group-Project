from datasets import load_dataset
import ollama
import json
from tqdm import tqdm 

# Configuration
dataset_name = "aqua_rat"
model_name = "wizard-math"
num_samples = 10  # Specify the number of samples to test

prompt_instruction = "You must indicate the correct answer by stating 'The correct answer is [LETTER]'. You must use that format."

# Load the dataset
dataset = load_dataset(dataset_name, split='test')

responses = []

# Use tqdm for the loading bar, wrapping the range function with tqdm
for i in tqdm(range(min(num_samples, len(dataset))), desc="Generating Responses"):  # Ensure not to exceed dataset length
    # Extracting the prompt and options
    prompt = dataset[i]['question']
    options = dataset[i]['options']
    correct = dataset[i]['correct']  # Correct answer
    options_text = ', '.join([f"{chr(65+j)}:{option}" for j, option in enumerate(options)])  # Formatting options with letters
    
    # Construct the prompt for OLLaMa
    enhanced_prompt = f"Question: {prompt}\nAnswer choices: {options_text}." + prompt_instruction

    # Generate a response using OLLaMa
    ollama_response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': enhanced_prompt}])
    response_text = ollama_response['message']['content']

    # Store the prompt, response, and correct answer
    responses.append({
        "prompt": enhanced_prompt.replace(prompt_instruction, ""),
        "response": response_text,
        "correct": correct
    })

# Save the responses to a JSON file
with open('responses.json', 'w') as f:
    json.dump(responses, f, indent=4)

print(f"Generated and saved {len(responses)} responses.")
