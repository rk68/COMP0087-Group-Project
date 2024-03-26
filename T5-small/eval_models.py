from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from datasets import load_dataset
import numpy as np
from functools import partial

def preprocess_aquarat(examples, tokenizer):
    questions_and_options = [
        f"question: {q} options: {opts[0]} {opts[1]} {opts[2]} {opts[3]} {opts[4]}." 
        for q, opts in zip(examples["question"], examples["options"])]

    correct_answers = [opts[ord(examples["correct"][i]) - ord('A')] for i, opts in enumerate(examples["options"])]

    input_encodings = tokenizer(questions_and_options, padding="max_length", truncation=True, max_length=512)
    target_encodings = tokenizer(correct_answers, padding="max_length", truncation=True, max_length=128)
    
    return {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "labels": target_encodings.input_ids,
        "inputs": questions_and_options,
        "answers": correct_answers
    }

def generate_answer(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def evaluate_model_accuracy(model, tokenizer, dataset):
    model.eval()  # Put model in evaluation mode
    correct = 0
    total = len(dataset)
    perc = 0.25
    for i, (input, answer) in enumerate(zip(dataset['input_ids'], dataset['answers'])):

        pred_answer = generate_answer(model, tokenizer, input)
        #print(pred_answer)
        #print(answer)
        # Compare predicted answer to the actual answer
        if pred_answer == answer:
            correct += 1
        
        if (i+1)/len(dataset) > perc:
            print(f'Testing... {np.round(perc*100, 2)}%')
            perc += 0.25
    print(f'Testing... 100.0%')
    accuracy = correct / total
    return accuracy

def main():
    base_model_path = 'google-t5/t5-small'
    base_model = T5ForConditionalGeneration.from_pretrained(base_model_path)
    full_finetuned_model = T5ForConditionalGeneration.from_pretrained('./models/full_finetune_t5-small_custom')
    lora_finetuned_model = T5ForConditionalGeneration.from_pretrained('./models/lora_finetune_t5-small_custom')
    kd_full_finetuned_model = T5ForConditionalGeneration.from_pretrained('./models/kd_full_finetune_t5-small_custom')
    tokenizer = T5Tokenizer.from_pretrained(base_model_path)

    test_data_name = 'aqua_rat'
    test_data = load_dataset(test_data_name, split='test')
    test_data = test_data.shuffle(seed=42).select(range(50))
    preprocess_with_tokenizer = partial(preprocess_aquarat, tokenizer=tokenizer)
    test_data_proc = test_data.map(preprocess_with_tokenizer, batched=True)

    # print('Testing T5-small on AquaRat...')
    # base_model_accuracy = evaluate_model_accuracy(base_model, tokenizer, test_data_proc)
    # print(f'T5_small AquaRat accuracy: {base_model_accuracy}')

    # print('\nTesting Full-finetuned model on AquaRat...')
    # fill_finetuned_model_accuracy = evaluate_model_accuracy(full_finetuned_model, tokenizer, test_data_proc)
    # print(f'Full-finetuned model AquaRat accuracy: {fill_finetuned_model_accuracy}')

    # print('\nTesting Lora-finetuned model on AquaRat...')
    # lora_finetuned_model_accuracy = evaluate_model_accuracy(lora_finetuned_model, tokenizer, test_data_proc)
    # print(f'Lora-finetuned model AquaRat accuracy: {lora_finetuned_model_accuracy}')

    print('\nTesting Knowledge Distillation Full-fintuned model on AquaRat...')
    kd_full_finetuned_model_accuracy = evaluate_model_accuracy(kd_full_finetuned_model, tokenizer, test_data_proc)
    print(f'Knowledge Distillation Full-fintuned model AquaRat accuracy: {kd_full_finetuned_model_accuracy}')

    # print(f'\nBase Accuracy: {np.round(base_model_accuracy, 4)}')
    # print(f'Full Accuracy: {np.round(fill_finetuned_model_accuracy, 4)}')
    # print(f'LORA Accuracy: {np.round(lora_finetuned_model_accuracy, 4)}')
    print(f'KD_full Accuracy: {np.round(kd_full_finetuned_model_accuracy, 4)}')

if __name__ == '__main__':
    main()

