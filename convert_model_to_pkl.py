import torch
from transformers import T5ForConditionalGeneration

# Load the model from the checkpoint
model_path = './models/checkpoint-245'  # Path to your checkpoint directory
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Save the model's state_dict in PKL format
pkl_save_path = './models/final_checkpoint.pkl'
with open(pkl_save_path, 'wb') as f:
    torch.save(model.state_dict(), f)  # Save only the model's state_dict in PKL format

print(f'Model weights saved in PKL format at {pkl_save_path}')
