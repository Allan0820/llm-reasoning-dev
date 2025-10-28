from transformers import  AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
import re 
import sympy
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()


df = pd.read_csv('fol_sympy_nl_16k.csv')
inputs = df['natural_language'].to_list()   # the input to the model is a natural lanugage statement 
outputs = df['sympy'].to_list()             # the output of the model is a sympy statement 
test_loader_tuple = zip(inputs, outputs)    # combining the data in the form of (input, output) to be tested
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
wandb_api = os.getenv("WANDB")
TOKEN = os.getenv("TOKEN")
login(token = TOKEN)


def prediction(checkpoint_path, pretrained_tokenizer): #evaluator function 
   
    models = os.listdir(checkpoint_path)
    model_load_path=[]
    model_nums = []
    predictions = None 

    for model in models:
        model_nums.append(int(model.split('-')[1]))
    model_nums.sort()
    print("Starting the predictions")
    
    for model_num in model_nums:
        # model_load_path.append(f'{os.getcwd()}' +  "/results/checkpoint-" + str(model_num))
        model_load_path.append(checkpoint_path + '/checkpoint-' + str(model_num))
    
    # print(model_load_path)
    # exit()
    
    for run_model in model_load_path:
        
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        model = AutoModelForCausalLM.from_pretrained(run_model)
        model.to('cuda')
        
        for input, output in test_loader_tuple:
            
            input_sentence_tokenized = tokenizer(input, return_tensors = 'pt').to('cuda')
            model.generation_config.cache_implementation = 'static'
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        
            with torch.no_grad():
                predictions = model.generate(**input_sentence_tokenized, max_new_tokens = 1)
             
                predicted_output_sentence = tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
                
                print("1] Actual ", output , "\n", "2] Predicted ", predicted_output_sentence)
            
            # model.forward = torch.compile(model.forward, mode = 'reduce-overhead', fullgraph = True )
            # input_ids = tokenizer(input, return_tensors = 'pt')
            # outputs = model.generate(**input_ids)
            # print(tokenizer.batch_decode(outputs, skip_special_tokens = True))
            # print("The predicted output is ")
        
        
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

# Start

#clear cache before you start 



prediction('./results/llama', 'meta-llama/Llama-3.2-1B')