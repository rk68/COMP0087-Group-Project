from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch
import pandas as pd
import sqlite3
import numpy as np
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from full_finetune import preprocess_custom



def get_dataset(num_samples):
    dataset_name = "./teacher_llm_dataset.db"
    conn = sqlite3.connect(dataset_name)
    query = "SELECT * FROM LLM_results"
    df_custom = pd.read_sql_query(query, conn)
    conn.close()
    training_dataset = Dataset.from_pandas(df_custom)
    training_dataset = training_dataset.shuffle(seed=42).select(range(num_samples))
    return training_dataset
    

def lora_finetune(model, dataset, tokenizer):
    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    #Select training parameters
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
        logging_steps=50,
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
    

    # # Define training args
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     auto_find_batch_size=True,
    #     learning_rate=1e-3, # higher learning rate
    #     num_train_epochs=5,
    #     logging_dir=f"{output_dir}/logs",
    #     logging_strategy="steps",
    #     logging_steps=50,
    #     save_strategy="no",
    #     report_to="tensorboard",
    # )
    # # Create Trainer instance
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=processed_dataset,
    # )


    #Train model
    print('Lora-finetuning Training')
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

    preprocess_with_tokenizer = partial(preprocess_custom, tokenizer=tokenizer)
    train_dataset_proc = train_dataset.map(preprocess_with_tokenizer, batched=True)
    print('Training data processed')

    lora_finetuned_model = lora_finetune(model, train_dataset_proc, tokenizer)

    output_path = './models/lora_finetune_' + model_name + '_' + dataset_name
    lora_finetuned_model.save_pretrained(output_path)

    



if __name__ == '__main__':
    main()
