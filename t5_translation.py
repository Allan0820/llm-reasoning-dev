#driver code for cross-attention based translation
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

TOKEN = os.getenv("TOKEN")
login(token = TOKEN)

ds = load_dataset("yuan-yang/MALLS-v0")

train_validation_split = ds['train'].train_test_split(0.2)

train_split = train_validation_split['train']
validation_split = train_validation_split['test']
test_split = ds['test']
print(train_split['FOL'])
exit()

torch.cuda.empty_cache()  # Clear GPU RAM before starting the training of the program

for model_name in config.MODEL_LIST:
    
   tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')
   model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
   
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   model.resize_token_embeddings(len(tokenizer))
    # Update the model config with the new pad token ID
   model.config.pad_token_id = tokenizer.pad_token_id
   # train_tokenized = tokenizer()
   train_tokenized = train_split.map(trainer.tokenize, batched= True, fn_kwargs={"tokenizer": tokenizer})
   valid_tokenized = validation_split.map(trainer.tokenize, batched = True, fn_kwargs={"tokenizer": tokenizer})
   test_tokenized = test_split.map(trainer.tokenize, batched= True, fn_kwargs={"tokenizer": tokenizer})
   
   trainer.train_model(model, train_tokenized, valid_tokenized, config.EPOCHS, model_name)
   
   del model, tokenizer
   torch.cuda.synchronize()  # Soft stop the GPU and ensure all processes finish
   torch.cuda.empty_cache()  # Clear GPU RAM before starting the next execution
  
   
   

   