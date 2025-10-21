import json
import matplotlib.pyplot as plt

# Load file
with open("./results/gemma3_1b/checkpoint-1195/trainer_state.json") as f:
    data = json.load(f)

log_history = data["log_history"]
steps, losses, grad_norms = [], [], []
eval_steps, eval_losses, eval_grad_norms = [],[],[]

for d in log_history:
    if d['step'] in steps:
        eval_steps.append(int(d['epoch']))
        eval_losses.append(d['eval_loss'])
        
        continue
    steps.append(d['step'])
    losses.append(d['loss'])
    grad_norms.append(d['grad_norm'])
    

# Plot loss
def plot_loss(x_val, y_val, title, x_label, y_label, saveas):
    plt.figure()
    plt.plot(x_val, y_val)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(saveas)
    print(title)



