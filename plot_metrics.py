
import json
import matplotlib.pyplot as plt

# Load training logs
with open('./results/checkpoint-600/trainer_state.json', 'r') as f:
    state = json.load(f)

log_history = state['log_history']

# Extract metrics
train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
steps = [entry['step'] for entry in log_history if 'loss' in entry]
eval_steps = [entry['step'] for entry in log_history if 'eval_loss' in entry]

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(steps, train_loss, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_curve.png")
plt.show()

# Plot Eval Loss
plt.figure(figsize=(10, 5))
plt.plot(eval_steps, eval_loss, color='orange', label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Validation Loss over Steps")
plt.legend()
plt.grid(True)
plt.savefig("eval_loss_curve.png")
plt.show()