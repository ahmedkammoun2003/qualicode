import io
import logging
import math
import os
import pprint
import sys
import time
import json
import pdb 
from tqdm import tqdm
from datetime import datetime

import transformers
import torch

from Datasets_codeT5.apps_dataset import APPSBaseDataset
from trainers.trainer_plan import Trainer_Plan
from transformers import Trainer  

import torch.multiprocessing

# Détection des GPU disponibles
torch.multiprocessing.set_sharing_strategy('file_system')
num_gpus = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {num_gpus} GPUs")

def run_training(args, train_data):
    if args.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
        model_path = args.model_path if args.model_path is not None else f"Salesforce/{args.model}"
        print(f"Loading model from {model_path}...")
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_path,
            tuning_mode=args.tuning_mode, 
            clone_pl_head=args.clone_pl_head
        )
        
        if args.clone_pl_head:
            print("Initializing Plan head with finetuned LM head...")
            lm_head_params = model.lm_head.weight.detach().numpy()
            model.pl_head.weight = torch.nn.Parameter(torch.tensor(lm_head_params))
    
    print(f'Finished loading model {args.model}')
    
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # Load checkpoint if it exists
    start_iteration = 0
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.save_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_iteration = checkpoint['iteration']
            print(f"Resuming from iteration {start_iteration}")
    
    train_data.start_iteration = start_iteration
    print(f"Starting main loop from iteration {start_iteration}")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True, 
        
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps * num_gpus,  

        learning_rate=5e-6,
        weight_decay=0.05,
        lr_scheduler_type='constant_with_warmup',

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,

        fp16=True,
        torch_compile=True,

        local_rank=-1,
    )
    
    trainer = Trainer_Plan(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tuning_mode=args.tuning_mode,
    )
    
    trainer.train()
    
    if args.local_rank in [-1, 0]:
        # Save final model and checkpoint
        trainer.save_model()
        final_checkpoint = {
            'iteration': start_iteration + args.epochs * len(train_data),
            'model_state_dict': model.state_dict(),
        }
        checkpoint_path = os.path.join(args.save_dir, "checkpoint.pkl")
        torch.save(final_checkpoint, checkpoint_path)
        
        # Save a separate final checkpoint
        model_save_path = os.path.join(args.save_dir, "final_checkpoint.pkl")
        with open(model_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)

def get_dataset(args): 
    fnames = os.listdir(args.train_path) 
    
    if args.db:
        fnames = fnames[:50]

    max_tokens = 512 if args.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] else 1024
    max_src_tokens = 600 if args.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] else -1
    
    train_data = APPSBaseDataset(
        dataroot=args.train_path, 
        problem_dirs=fnames,
        model=args.model,
        max_tokens=max_tokens,
        max_src_tokens=max_src_tokens,
        sample_mode=args.sample_mode
    )
    return train_data

def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_data = get_dataset(args)
    
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))
    
    run_training(args, train_data)

if __name__ == "__main__":
    from configs.train_codet5_configs import *
    main(args)
