import transformers 
from huggingface_hub import notebook_login
from dotenv import load_dotenv 
import os
load_dotenv()

TOKEN = os.getenv("TOKEN") 

checkpoints_dirs = os.listdir('./results/')

for x in checkpoints_dirs:
    model = os.listdir(x+'/')
    for y in model:
        print(y)
        