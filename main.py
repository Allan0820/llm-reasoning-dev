#driver code 
import config 
import trainer
import gc
import pandas as pd 
import torch
import os 
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
load_dotenv()

# os.environ['PYTORCH_ALLOC_CONF'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'False' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

TOKEN = os.getenv("TOKEN")
login(token = TOKEN)

data_files = {
    "train": "fol_sympy_nl_16k.csv",
    "test": "fol_sympy_nl.csv"
}

dataset = load_dataset("csv", data_files = data_files)
dataset = dataset.remove_columns('id')

train_val_split = dataset["train"].train_test_split(test_size = 0.05, shuffle=True) #small validation set should suffice 


train_split = train_val_split['train']
val_split = train_val_split['test']
test_split = dataset["test"]

train_split.shard(num_shards=10, index=0)

for model_name in config.MODEL_LIST:
    
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
   
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   train_tokenized = train_split.map(trainer.tokenize, batched= True, fn_kwargs={"tokenizer": tokenizer})
   valid_tokenized = val_split.map(trainer.tokenize, batched = True, fn_kwargs={"tokenizer": tokenizer})
   test_tokenized = test_split.map(trainer.tokenize, batched= True, fn_kwargs={"tokenizer": tokenizer})
#    torch.cuda.synchronize()  # Soft stop the GPU and ensure all processes finish
#    torch.cuda.empty_cache()  # Clear GPU RAM before starting the next execution
   trainer.train_model(model, train_tokenized, valid_tokenized, config.EPOCHS)
   
   del model, tokenizer
   torch.cuda.synchronize()  # Soft stop the GPU and ensure all processes finish
   torch.cuda.empty_cache()  # Clear GPU RAM before starting the next execution
  
   
   

   