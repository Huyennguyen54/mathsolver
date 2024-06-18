mport json
import os
import torch
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

data = load_dataset('hllj/vi_grade_school_math_mcq')

def generate_prompt(question, choices, explanation):
    return f"""
<|im_start|>system
Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.

<|im_start|>user
### Câu hỏi:
{question}
### Các lựa chọn:
{choices}
### Câu trả lời:

<|im_start|>assistant
{explanation}
""".strip()

def generate_and_tokenize_prompt(question, choices, explanation):
    full_prompt = generate_prompt(question, choices, explanation)
    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )

    return tokenized_full_prompt
training_samples = []
for sample in tqdm(data['train']):
    for quest in sample['problems']:
        choices = quest['choices']
        explanation = quest['explanation'].strip()
        question = quest['question']

        if explanation == '' or question == '' or choices == []:
            continue

        try:
            question = question.split('\n \n')[1].strip()
        except:
            continue

        choices = '\n'.join(choices)
        training_sample = generate_and_tokenize_prompt(
            question, choices, explanation
        )

        training_samples.append(training_sample)
choices_data = Dataset.from_list(training_samples)