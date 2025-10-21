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
    
# Extract values

# Plot loss
plt.figure()
plt.plot(steps, losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("Training_Loss.png")
print("plotted loss")

# Plot grad norm
plt.figure()
plt.plot(steps, grad_norms)
plt.title("Gradient Norm")
plt.xlabel("Step")
plt.ylabel("Grad Norm")
plt.savefig('Gradient_Norm.png')
print("plotted gradnorm")

# Evaluation losses 

plt.figure()
plt.plot(eval_steps, eval_losses)
plt.title("raw_evaluation_loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig('raw_evaluation_loss.png')
print("plotted raw_evaluation_loss")