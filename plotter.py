
import json
import os
import re 
import sympy
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()

from transformers import  AutoModelForCausalLM, AutoTokenizer
import pandas as pd

df = pd.read_csv('fol_sympy_nl_16k.csv')
inputs = df['natural_language'].to_list()
outputs = df['sympy'].to_list()
test_loader_tuple = zip(inputs, outputs)

wandb_api = os.getenv("WANDB")
TOKEN = os.getenv("TOKEN")
login(token = TOKEN)


def prediction(checkpoint_path): #evaluator function 
   
    models = os.listdir(checkpoint_path)
    model_load_path=[]
    model_nums = []
    predictions = None 

    for model in models:
        model_nums.append(int(model.split('-')[1]))
    model_nums.sort()

    for model_num in model_nums:
        # model_load_path.append(f'{os.getcwd()}' +  "/results/checkpoint-" + str(model_num))
        model_load_path.append('./results/checkpoint-' + str(model_num) + "/")
    
    
    for run_model in model_load_path:
        for input, output in test_loader_tuple:
                    
            print(input+'------'+output)
            print(run_model)
            tokenizer = AutoTokenizer.from_pretrained(run_model, local_files_only = True, trust_remote_code = True)
            model = AutoModelForCausalLM.from_pretrained(run_model, local_files_only = True, trust_remote_code = True)
            exit()
            input_sentence_tokenized = tokenizer(input, return_tensors = 'pt')
            exit()
            with torch.no_grad():
                predictions = model.generate(**input_sentence_tokenized)
                
            print('The predictions are')
            print(predictions)
            # predicted_output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(predicted_output_sentence)
    
        
    matched_true = 0
    matched_false = 0
         
def plot_loss(model_path, saveas):
    
    
    
    with open(checkpoint_path+'/trainer_state.json') as f:
       data = json.load(f)

    log_history = data["log_history"]
    steps, losses, grad_norms = [], [], []
    eval_steps, eval_losses = [],[]
    raw_test_epoch, raw_test_error = prediction(checkpoint_path)
    sympy_test_epoch, sympy_test_error = prediction(checkpoint_path)

    for d in log_history:
        if d['step'] in steps:
            eval_steps.append(int(d['epoch']))
            eval_losses.append(d['eval_loss'])
            continue
        
        steps.append(d['step'])
        losses.append(d['loss'])
        grad_norms.append(d['grad_norm'])
    
    plt.figure()
    plt.plot(x_val, y_val)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(saveas)
    print(title)

print("The plotting function begins")

# plot_loss(steps, y_val, "./results/gemma3_1b/checkpoint-1195/trainer_state.json", x_label, y_label, saveas)

prediction("./results/gemma3_1b")