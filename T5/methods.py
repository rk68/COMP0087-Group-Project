from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import sqlite3
import pandas as pd
import re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
import numpy as np

def get_dataset(num_samples, seed):
    dataset_name = "../data/teacher-data/teacher_llm_dataset_2000.db"
    conn = sqlite3.connect(dataset_name)
    query = "SELECT * FROM LLM_results"
    df_custom = pd.read_sql_query(query, conn)
    conn.close()
    training_dataset = Dataset.from_pandas(df_custom)
    training_dataset = training_dataset.shuffle(seed=seed).select(range(num_samples))
    return training_dataset

def preprocess_custom(examples, tokenizer):
    options = [re.findall(r'"([^"]+)"', text) for text in examples["options"]]

    questions_and_options = [
        f"question: {q} options: {opts[0]} {opts[1]} {opts[2]} {opts[3]} {opts[4]}" 
        for q, opts in zip(examples["question"], options)]

    correct_answers = [opts[ord(examples["correct"][i]) - ord('A')] for i, opts in enumerate(options)]

    input_encodings = tokenizer(questions_and_options, padding="max_length", truncation=True, max_length=512)
    target_encodings = tokenizer(correct_answers, padding="max_length", truncation=True, max_length=128)
    
    return {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "labels": target_encodings.input_ids,
        "inputs": questions_and_options,
        "answers": correct_answers
    }

def lora_finetune(model, dataset, tokenizer, device):
    # Move model to appropriate device
    model.to(device)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    # Prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # We want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    # Select training parameters
    training_args = TrainingArguments(
        output_dir="./checkpoints/lora_checkpoints",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=25,
        #load_best_model_at_end=True,
        #gradient_accumulation_steps=2,  # Accumulate gradients for 2 steps
        #max_grad_norm=1.0,  # Clip gradients to have a maximum norm of 1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # eval_dataset=processed_eval_dataset, # If you have an evaluation dataset
    )

    # Train model
    print('Lora-finetuning Training')
    trainer.train()

    print('Training complete')
    return model
    
def preprocess_custom_kd(examples, tokenizer):
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


def full_finetune(model, dataset, device):
    # Move model to appropriate device
    model.to(device)

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