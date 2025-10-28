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
from sentence_transformers import SentenceTransformer
load_dotenv()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv('fol_sympy_nl.csv')
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
        model_load_path.append(checkpoint_path + '/checkpoint-' + str(model_num))
    
    for run_model in model_load_path:
        
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        model = AutoModelForCausalLM.from_pretrained(run_model)
        model.to('cuda')
        actual_inputs, predicted_outputs, cosine_similarity, ground_truth = [], [],[], []
        counter = 0
        
        for input, output in test_loader_tuple:
            
            input_sentence_tokenized = tokenizer(input, return_tensors = 'pt').to('cuda')
            model.generation_config.cache_implementation = 'static'
            model.generation_config.pad_token_id = tokenizer.eos_token_id
            
            with torch.no_grad():
                predictions = model.generate(**input_sentence_tokenized, max_new_tokens = 1)
                predicted_output_sentence = tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
                actual_inputs.append(input) ; predicted_outputs.append(predicted_output_sentence) ; ground_truth.append(output)
                
                
                sentences = [output,predicted_output_sentence]
                # in_emb, out_emb = sentence_model.encode(output), sentence_model.encode(list(predicted_output_sentence))
                # cs = torch.nn.CosineSimilarity(dim = 1, eps = 1e-8)
                embeddings = sentence_model.encode(sentences)
                similarity_score = sentence_model.similarity(embeddings, embeddings)
                cosine_similarity.append(similarity_score)
                        
                print(similarity_score.numpy()[0][1])
             
                counter +=1
                print(counter)
                
                print("1] Actual ", output , "\n", "2] Predicted ", predicted_output_sentence)
                
        assert len(actual_inputs) == len(predicted_outputs) == len(cosine_similarity)
        
        data = {
            'Inputs' : actual_inputs,
            'Predictions' : predicted_outputs,
            'Cosine_Similarity' : cosine_similarity
        }
        
        df=pd.DataFrame(data)
        df.to_csv('./results/model_outputs/' + str(model_load_path.index(run_model)) +'.csv')
        
   
    matched_true = 0
    matched_false = 0
         
# def plot_loss(model_path, saveas):
    
#     with open(checkpoint_path+'/trainer_state.json') as f:
#        data = json.load(f)

#     log_history = data["log_history"]
#     steps, losses, grad_norms = [], [], []
#     eval_steps, eval_losses = [],[]
#     raw_test_epoch, raw_test_error = prediction(checkpoint_path)
#     sympy_test_epoch, sympy_test_error = prediction(checkpoint_path)

#     for d in log_history:
#         if d['step'] in steps:
#             eval_steps.append(int(d['epoch']))
#             eval_losses.append(d['eval_loss'])
#             continue
        
#         steps.append(d['step'])
#         losses.append(d['loss'])
#         grad_norms.append(d['grad_norm'])
    
#     plt.figure()
#     plt.plot(x_val, y_val)
#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.savefig(saveas)
#     print(title)

print("The plotting function begins")

prediction('./results/llama', 'meta-llama/Llama-3.2-1B')