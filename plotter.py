
'''
Directory structure should be as follows 

IF
Driver Program --> plotter.py -> root directory 
THEN
Result Folder with trained models --> ./results/<model_name>/<model_checkpoint_folder>
AND
Prediction Folder for outputs of the trained models --> ./results/<model_name>/<model_outputs>/

'''

from transformers import  AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
import pandas as pd
import json
import numpy
import os
import re 
import sympy
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
load_dotenv()
# wandb.login(TOKEN = os.getenv('WANDB'))

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
torch.cuda.empty_cache()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
wandb_api = os.getenv("WANDB")
TOKEN = os.getenv("TOKEN")
login(token = TOKEN)

ds = load_dataset("yuan-yang/MALLS-v0")
test = ds['test']
test_df = test.to_pandas()
test_loader_tuple = list(zip(list(test_df['NL']),list(test_df['FOL']))) #converting back to a list to remove the iterator prop from zip

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
        model_load_path.append(checkpoint_path + '/checkpoint-' + str(model_num))
    Epoch = 0 ; epoch_counter = [] ; avg_cosine_similarity = []

    name_version = os.path.basename(os.path.normpath(pretrained_tokenizer))
    
    '''========================================================================='''
    try:
        os.system(f'mkdir ./results/model_outputs/{name_version} > /dev/null') 
    except: 
        print('folder exists!')
    for run_model in model_load_path:
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        model = AutoModelForCausalLM.from_pretrained(run_model)
        actual_inputs, predicted_outputs, cosine_similarity, ground_truth, actual_outputs = [], [],[], [], []
        counter = 0
        # print("Finished with outer loop")
        for input, output in test_loader_tuple:
            # print("Starting with the inner loop")
            
            input_sentence_tokenized = tokenizer(input, return_tensors = 'pt').to('cuda')
            model.generation_config.cache_implementation = 'static'
            model.generation_config.pad_token_id = tokenizer.eos_token_id
            model.to('cuda')
            with torch.no_grad():
                predictions = model.generate(**input_sentence_tokenized)
                predicted_output_sentence = tokenizer.batch_decode(predictions, skip_special_tokens=True, max_new_tokens =1)[0]
                actual_inputs.append(input) ; predicted_outputs.append(predicted_output_sentence) ; ground_truth.append(output)
                actual_outputs.append(output)
                
                sentences = [output,predicted_output_sentence]
                embeddings = sentence_model.encode(sentences)
                similarity_score = sentence_model.similarity(embeddings, embeddings)
                cosine_similarity.append(similarity_score.numpy()[0][1])
                counter +=1
                print(counter)   
                #print("1] Actual ", output , "\n", "2] Predicted ", predicted_output_sentence)
                
        '''========================================================================='''     
           
        assert len(actual_inputs) == len(predicted_outputs) == len(cosine_similarity) == len(actual_outputs)
        
        data = {
            'Inputs' : actual_inputs,
            'Outputs_GT' : actual_outputs,
            'Predictions' : predicted_outputs,
            'Cosine_Similarity' : cosine_similarity
        }
        avg_cosine_similarity.append(sum(cosine_similarity)/len(cosine_similarity))
        datafinal=pd.DataFrame(data)
      
        Epoch +=1
        epoch_counter.append(Epoch)
        datafinal.to_csv(f'./results/model_outputs/{name_version}' + f"Epoch_{Epoch}__"+ name_version +'.csv')
        print("saved_df of ", run_model) 
        
        '''=========================== Below is the plotting function for the training and eval curves (cross-entropy loss) =================='''
        
        with open(run_model+'/trainer_state.json') as f:
             data = json.load(f)

        log_history = data["log_history"]
        train_epochs, train_losses= [], []
        steps, losses = [], []
        eval_epochs, eval_losses = [], []

        for d in log_history:
            if d['step'] in steps:
                eval_epochs.append(int(d['epoch']))
                train_epochs = eval_epochs
                eval_losses.append(d['eval_loss'])
                train_losses.append(sum(losses)/len(losses))
                losses = []
                continue
            steps.append(d['step'])
            losses.append(d['loss'])
        
        
        plt.figure()
        plt.plot(train_epochs, train_losses)
        plt.title(name_version + " Training Curves")
        plt.ylabel("Training Loss")
        plt.xlabel("Training Epochs")
        plt.savefig(f'./results/model_outputs/{name_version}' + name_version + '_training.jpg')
        
        plt.figure()
        plt.plot(eval_epochs, eval_losses)
        plt.title(name_version + " Validation Curves")
        plt.ylabel("Validation Loss")
        plt.xlabel("Validation Epochs")
        plt.savefig(f'./results/model_outputs/{name_version}' + name_version + '_validation.jpg')
        
        plt.figure()
        plt.plot(epoch_counter, avg_cosine_similarity)
        plt.title(name_version + " Testing Curves")
        plt.ylabel("Testing Loss")
        plt.xlabel("Testing Epochs")
        plt.savefig(f'./results/model_outputs/{name_version}' + name_version + '_testing.jpg')
      
        
    Epoch = 0
 
    print("Epoch Number Finished ", Epoch)

print("LLAMA 1B")
# prediction('./results/llama', 'meta-llama/Llama-3.2-1B')
# print("LLAMA 3B")
prediction('./results/llama3b', 'meta-llama/Llama-3.2-3B')
# print("GEMMA3 1B")
# prediction('./results/gemma3', 'google/gemma-3-270m')
# print("ROBERTA 1B")
# prediction('./results/roberta', 'nyu-mll/roberta-base-1B-3')