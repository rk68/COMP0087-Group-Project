from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration
)
import torch
import numpy as np
from functools import partial
import argparse
from methods import (
    get_dataset,
    preprocess_custom,
    full_finetune,
    preprocess_custom_kd,
    lora_finetune
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-small", help="Name of the model to train")
    parser.add_argument("--model_path", type=str, default="google-t5/t5-small", help="Path of model to train")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset to train on")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples used for training")
    parser.add_argument("--lora", action="store_true", default=False, help="If set, overwrite the 4D-Human tracklets.")
    parser.add_argument('--kd', action='store_true', default=False, help='If set, save meshes to disk.')
    parser.add_argument('--seed', type=int, action='store', default=None, help='If set, seed.')
    args = parser.parse_args()

    seed = args.seed

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = args.model_name
    model_path = args.model_path
    num_samples = args.num_samples
    dataset_name = args.dataset

    agrs_print = f'\nFinetuning {model_name} on {dataset_name} dataset, {num_samples} samples\n'
    if args.lora:
        agrs_print += 'Lora Finetuning'
    else:
        agrs_print += 'Full Finetuning'
    
    if args.kd:
        agrs_print += ', Knowledge Distillation'
    
    print(agrs_print + '\n')
    
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    print('Model and tokenizer loaded')

    if dataset_name == 'custom':
        train_dataset = get_dataset(num_samples, args.seed)
    
    print('Training dataset loaded')
    if args.kd:
        preprocess_with_tokenizer = partial(preprocess_custom_kd, tokenizer=tokenizer)
    else:
        preprocess_with_tokenizer = partial(preprocess_custom, tokenizer=tokenizer)
        
    train_dataset_proc = train_dataset.map(preprocess_with_tokenizer, batched=True)
    print('Training data processed')

    if args.lora:
        finetuned_model = lora_finetune(model, train_dataset_proc, tokenizer, device)
    else:
        finetuned_model = full_finetune(model, train_dataset_proc, device)

    output_path = './' + model_name + '/'
    if args.kd:
        output_path += 'kd_'
    if args.lora:
        output_path += 'lora_finetune_' + model_name + '_' + dataset_name + str(num_samples)
    else:
        output_path += 'full_finetune_' + model_name + '_' + dataset_name + str(num_samples)
    finetuned_model.save_pretrained(output_path)

if __name__ == '__main__':
    main()
