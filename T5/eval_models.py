from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM
)
from datasets import load_dataset
import numpy as np
from functools import partial
import argparse
from peft import LoraModel
from collections import Counter

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
    model_answers = []
    letters = []
    for i, (input, answer) in enumerate(zip(dataset['input_ids'], dataset['answers'])):

        pred_answer = generate_answer(model, tokenizer, input)
        #print(pred_answer)
        #print(answer)
        # Compare predicted answer to the actual answer
        if pred_answer:
            letters.append(pred_answer[0])
        model_answers.append((tokenizer.decode(input, skip_special_tokens=True), answer, pred_answer))
        if pred_answer == answer:
            correct += 1
        
        if (i+1)/len(dataset) > perc:
            print(f'Testing...  {np.round(perc*100, 2)}%')
            perc += 0.25
    print(f'Testing... 100.0%')
    accuracy = correct / total
    return accuracy, model_answers, letters

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="full_finetune_t5-small_custom1000", help="Path of model to test")
    parser.add_argument("--base_model", type=str, default="t5-small", help="Base model")
    parser.add_argument("--base_model_path", type=str, default="google-t5/t5-small", help="Base model path")
    parser.add_argument("--dataset", type=str, default="aqua_rat", help="Dataset to test on")
    parser.add_argument("--print_answers",action='store_true', default="aqua_rat", help="If set, print example answers")
    args = parser.parse_args()

    base_model_path = args.base_model_path
    if args.model_name == 't5-small' or args.model_name == 't5-base' or args.model_name == 'flan-t5-base':
        model_name = base_model_path
    else:
        model_name = './'+ args.base_model+'/' + args.model_name
    dataset = args.dataset
    print(model_name)
    model =  T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(base_model_path)
    test_data = load_dataset(dataset, split='test')

    preprocess_with_tokenizer = partial(preprocess_aquarat, tokenizer=tokenizer)
    test_data_proc = test_data.map(preprocess_with_tokenizer, batched=True)

    print('\nTesting ' + args.model_name + ' on ' + dataset)
    accuracy, model_answers, letters = evaluate_model_accuracy(model, tokenizer, test_data_proc)

    print(f'\nAccuracy: {np.round(accuracy, 4)}')

    print(Counter(letters))

    for i in range(20, 25):
        case = model_answers[i]
        print(f'\nQuestion: {case[0]}')
        print(f'Correct Answer: {case[1]}')
        print(f'Model Answer: {case[2]}')

if __name__ == '__main__':
    main()

