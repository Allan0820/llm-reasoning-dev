#driver code 
import config 

import pandas as pd 
import torch
import os 
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset 
from huggingface_hub import login
import trainer
load_dotenv()
TOKEN = os.getenv("TOKEN")
login(token = TOKEN)

data_files = {
    "train": ""
}

for model_name in MODEL_LIST:

    tokenizer = AutoTokenizer.load_pretrained(model_name)
    model = AutoModelForCausalLM.load_pretrained(model_name)
      

print("initialized all the variables")

