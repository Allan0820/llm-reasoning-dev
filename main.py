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

print("hello")

