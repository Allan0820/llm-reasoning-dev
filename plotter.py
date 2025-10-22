import json
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

wandb_api=os.getenv("WANDB")
print(wandb_api)

def prediction(checkpoint_path): #evaluator function 
    #for every item in the dataset, check the ground truth and compare prediction
    models = os.listdir(checkpoint_path)
    models.sort()
    for model in models:
        
    pass 


    
def plot_loss(model_path, saveas):
    
    
    
    with open(checkpoint_path+'/trainer_state.json') as f:
       data = json.load(f)

    log_history = data["log_history"]
    steps, losses, grad_norms = [], [], []
    eval_steps, eval_losses = [],[]
    raw_test_epoch, raw_test_error = [], []
    sympy_test_epoch, sympy_test_error = [], []

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

plot_loss(steps, y_val, "./results/gemma3_1b/checkpoint-1195/trainer_state.json", x_label, y_label, saveas)

