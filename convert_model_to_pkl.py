import torch
from transformers import T5ForConditionalGeneration
import os

# Load the checkpoint first
model_path = './models/checkpoint-14320'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device)

# Initialize base model
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

# Add pl_head layer
model.pl_head = torch.nn.Linear(in_features=model.lm_head.in_features,
                               out_features=model.lm_head.out_features,
                               bias=False)

# Load the checkpoint weights including pl_head
model.load_state_dict(checkpoint_dict)
model.to(device)

# Save the model's state_dict in PKL format
pkl_save_path = './models/final_checkpoint.pkl'
torch.save(model.state_dict(), pkl_save_path)

print(f'Model weights saved in PKL format at {pkl_save_path}')
