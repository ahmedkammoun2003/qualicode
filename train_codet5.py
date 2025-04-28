import os
import json
import logging
import pprint
import torch
import transformers
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime
from codebleu import calc_codebleu

from Datasets_codeT5.apps_dataset import APPSBaseDataset
from trainers.trainer_plan import Trainer_Plan

# On dit à PyTorch comment partager les données quand on utilise plusieurs processus
torch.multiprocessing.set_sharing_strategy('file_system')

# On dit qu’on veut utiliser le GPU numéro 1 pour aller plus vite
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Remove the CUDA_VISIBLE_DEVICES setting to allow all GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # Comment out or remove this line

def setup_distributed(args):
    """Setup distributed training"""
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))

    if args.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def run_training(args, train_data, val_data, test_data):
    """
    🏋️‍♂️ Ici on entraîne le modèle avec nos données.
    Ensuite, on regarde s'il a bien appris avec CodeBLEU sur les données de validation et de test.
    """
    model = load_model(args)
    model.to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True
        )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type='constant_with_warmup',
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,
        local_rank=args.local_rank,
        ddp_backend='nccl',
    )

    trainer = Trainer_Plan(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data)

    print("Starting training...")
    trainer.train()

    if args.local_rank == 0:
        print("Saving final model checkpoint...")
        checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(checkpoint_dir)
        args.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "final_checkpoint.pkl"))

        # Evaluate on both validation and test sets
        print("Evaluating on validation set...")
        val_score = evaluate_with_codebleu(model, val_data, args)
        print("Evaluating on test set...")
        test_score = evaluate_with_codebleu(model, test_data, args)

        # Save both scores
        with open(os.path.join(args.save_dir, "codebleu_scores.json"), 'w') as f:
            json.dump({
                "validation_CodeBLEU": val_score,
                "test_CodeBLEU": test_score
            }, f, indent=4)

def main(args):
    """
    🚀 C'est la fonction principale qui lance tout le processus.
    Elle prépare les données, entraîne le modèle, puis l'évalue.
    """
    # Setup distributed training
    args = setup_distributed(args)
    print(pprint.pformat(vars(args)))

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    # Get all three datasets
    train_data, val_data, test_data = get_dataset(args)

    # Run training with all three datasets
    run_training(args, train_data, val_data, test_data)


# 🧁 Si on lance ce fichier tout seul (et pas importé), alors on exécute main()
if __name__ == "__main__":
    from configs.train_codet5_configs import get_args
    args = get_args()
    main(args)
