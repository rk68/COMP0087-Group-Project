from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import torch
import sqlite3
import pandas as pd
import sqlite3
import re
import numpy as np
from functools import partial


def get_dataset(num_samples):
    dataset_name = "./teacher_llm_dataset.db"
    conn = sqlite3.connect(dataset_name)
    query = "SELECT * FROM LLM_results"
    df_custom = pd.read_sql_query(query, conn)
    conn.close()
    training_dataset = Dataset.from_pandas(df_custom)
    training_dataset = training_dataset.shuffle(seed=42).select(range(num_samples))
    return training_dataset
    
def preprocess_knowledge_distillation(examples, tokenizer):
    options = [re.findall(r'"([^"]+)"', text) for text in examples["options"]]

    reasonings = ["\n\n".join([line for line in text.strip().split("\n\n") if "The answer is:" not in line]) for text in examples["LLMs_rationale"]]

    questions_options_reasoning = [
        f"question: {q} options: {opts[0]} {opts[1]} {opts[2]} {opts[3]} {opts[4]}, {reasoning}" 
        for q, opts, reasoning in zip(examples["question"], options, reasonings)]

    correct_answers = [opts[ord(examples["correct"][i]) - ord('A')] for i, opts in enumerate(options)]

    input_encodings = tokenizer(questions_options_reasoning, padding="max_length", truncation=True, max_length=512)
    target_encodings = tokenizer(correct_answers, padding="max_length", truncation=True, max_length=128)
    
    return {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "labels": target_encodings.input_ids,
        "inputs": questions_options_reasoning,
        "answers": correct_answers
    }


def full_finetune(model, dataset):
    #Select training parameters
    training_args = TrainingArguments(
        output_dir="./checkpoints/kd_full_checkpoints",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,
        #load_best_model_at_end=True,
        #gradient_accumulation_steps=2,  # Accumulate gradients for 2 steps
        #max_grad_norm=1.0,  # Clip gradients to have a maximum norm of 1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=processed_eval_dataset, # If you have an evaluation dataset
    )

    #Train model
    print('Full-finetuning Training')
    trainer.train()

    print('Training complete')
    return model

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    model_name = 't5-small'
    model_path = 'google-t5/t5-small'
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    print('Model and tokenizer loaded')

    num_samples = 1000
    dataset_name = 'custom'
    train_dataset = get_dataset(num_samples)
    print('Training dataset loaded')

    preprocess_with_tokenizer = partial(preprocess_knowledge_distillation, tokenizer=tokenizer)
    train_dataset_proc = train_dataset.map(preprocess_with_tokenizer, batched=True)
    print('Training data processed')

    full_finetuned_model = full_finetune(model, train_dataset_proc)

    output_path = './models/kd_full_finetune_' + model_name + '_' + dataset_name
    full_finetuned_model.save_pretrained(output_path)

    



if __name__ == '__main__':
    main()
