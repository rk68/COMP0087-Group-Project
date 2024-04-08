## COMP0087 Project: Advancing Mathematical Reasoning in Small Language Models

**Abstract:** This study investigates the use of parameter-efficient fine-tuning (PEFT) techniques to distil complex mathematical reasoning ability from Large Language Models (LLMs) into Small Language Models (SLMs). The AQuA-RAT dataset is used to generate teacher rationales (soft labels) which are then used to fine-tune several SLMs using full fine-tuning, low-rank adaptation and quantisation methods. Prompting techniques such as few-shot and Chain-of-Thought (CoT) for in-context learning are also tested. Experimental results demonstrate that knowledge distillation (KD) on LLM-generated rationales improves performance for both full fine-tuning (FFT) and PEFT in scenarios where these generally underperform. Furthermore, KD with PEFT enables comparable performance to fine-tuning on the original human-annotated dataset, with improvements under few-shot and CoT prompting. Hence, these findings suggest that PEFT and KD may be used to teach small and/or quantised models complex reasoning tasks without expensive human annotation. Further research, though, is required to test the effect of catastrophic forgetting and the ability to generalise to other adjacent reasoning tasks.

This repository contains the code and example results for our project. This includes code for fine-tuning:

- T5-Small (60M)
- T5-Base (220M)
- Phi 1.5 (1.3B)


We also provide our LLM-generated (using quantized Gemma-7B) mathematical rationales within the data folder.


All Phi 1.5 fine-tuned variants are available here: https://huggingface.co/rk68
