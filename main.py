#driver code 
import config 
import trainer

import pandas as pd 
import torch
import os 
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
load_dotenv()


TOKEN = os.getenv("TOKEN")
login(token = TOKEN)

data_files = {
    "train": "fol_sympy_nl_16k.csv",
    "test": "fol_sympy_nl.csv"
}

dataset = load_dataset("csv", data_files = data_files)
dataset = dataset.remove_columns('id')

train_val_split = dataset["train"].train_test_split(test_size = 0.1, shuffle=True)


train_split = train_val_split['train']
val_split = train_val_split['test']
test_split = dataset["test"]


for model_name in config.MODEL_LIST:
    
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   
   
   train_tokenized = tokenizer(
        list(train_split),
        truncation = True, 
        max_length = 128,
        padding = True,
        return_tensors = 'pt'
    )
   test_tokenized = tokenizer(
        list(train_split['sympy']),
        truncation = True, 
        max_length = 128,
        padding = True,
        return_tensors = 'pt'
    )
   
   
   
   
#    FOL = tokenizer(
#         x['sympy'],
#         truncation = True,
#         padding = True
#     )
    
    