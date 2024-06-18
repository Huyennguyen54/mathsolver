# -*- coding: utf-8 -*-
## 1. Install and import necessary libaries
"""

!pip install -q -U bitsandbytes
!pip install -q -U datasets
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U loralib
!pip install -q -U einops

import json
import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers

from pprint import pprint
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""## 2. Sign in to huggingface"""

notebook_login()

"""## 3. Load pretrained LLM"""

MODEL_NAME = "vilm/vinallama-7b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

"""## 4. Test pretrained model performance (make prediction)"""

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

prompt = """
<|im_start|>system
Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.

<|im_start|>user
### Câu hỏi:
Số gồm 1 đơn vị và 3 chục đọc là :
### Các lựa chọn:
A. 30
B. 31
C. 20
D. 21
### Câu trả lời:

<|im_start|>assistant
""".strip()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from IPython.display import Javascript
# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 500})'''))
# 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# encoding = tokenizer(prompt, return_tensors="pt").to(device)
# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids=encoding.input_ids,
#         attention_mask=encoding.attention_mask,
#         generation_config=generation_config
#     )
# 
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""## 5. Fine-tuning LLM

### 5.1. Prepare dataset
"""

data = load_dataset('hllj/vi_grade_school_math_mcq')

data

type(data)

data["train"]

data["train"][5]['problems'][0]

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

"""### 5.2. Training"""

training_args = transformers.TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=4,
      num_train_epochs=1,
      learning_rate=2e-4,
      fp16=True,
      save_total_limit=3,
      logging_steps=1,
      output_dir="experiments",
      optim="paged_adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=choices_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

"""### 5.3. Test prediction"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# prompt = """
# <|im_start|>system
# Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.
# 
# <|im_start|>user
# ### Câu hỏi:
# Số gồm 1 đơn vị và 2 chục đọc là :
# ### Các lựa chọn:
# A. 20
# B. 21
# C. 30
# D. 31
# ### Câu trả lời:
# 
# <|im_start|>assistant
# """.strip()
# 
# encoding = tokenizer(prompt, return_tensors="pt").to(device)
# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids=encoding.input_ids,
#         attention_mask=encoding.attention_mask,
#         generation_config=generation_config
#     )
# 
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""## 6. Save model to huggingface"""

model.save_pretrained("trained-model")

PEFT_MODEL = "thangduong0509/vinallama-peft-7b-math-solver"

model.push_to_hub(
    PEFT_MODEL, use_auth_token=True
)

"""## 7. Inference"""

PEFT_MODEL = "thangduong0509/vinallama-peft-7b-math-solver"

config = PeftConfig.from_pretrained(PEFT_MODEL)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = model.config.pad_token_id
generation_config.eos_token_id = model.config.eos_token_id

# Commented out IPython magic to ensure Python compatibility.
# %%time
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# prompt = """
# <|im_start|>system
# Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.
# 
# <|im_start|>user
# ### Câu hỏi:
# Số gồm 1 đơn vị và 3 chục đọc là :
# ### Các lựa chọn:
# A. 30
# B. 31
# C. 20
# D. 21
# ### Câu trả lời:
# 
# <|im_start|>assistant
# """.strip()
# 
# encoding = tokenizer(prompt, return_tensors="pt").to(device)
# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids=encoding.input_ids,
#         attention_mask=encoding.attention_mask,
#         generation_config=generation_config
#     )
# 
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))