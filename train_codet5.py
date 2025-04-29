import os
import json
import pprint
import torch
import logging
from tqdm import tqdm
from datetime import datetime

from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration
from Datasets_codeT5.apps_dataset import APPSBaseDataset
from trainers.trainer_plan import Trainer_Plan

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable multi-GPU support
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both GPU 0 and 1

def get_dataset(args):
    fnames = os.listdir(args.train_path)
    fnames.sort()

    if args.db:
        fnames = fnames[:50]
    
    # 70/20/10 split
    n = len(fnames)
    train_fnames = fnames[:int(0.7 * n)]
    val_fnames = fnames[int(0.7 * n):int(0.9 * n)]
    test_fnames = fnames[int(0.9 * n):]

    max_tokens = 512 if 'codet5' in args.model else 1024
    max_src_tokens = 600 if 'codet5' in args.model else -1

    logger.info(f"Train: {len(train_fnames)} samples, Val: {len(val_fnames)}, Test: {len(test_fnames)}")

    train_data = APPSBaseDataset(args.train_path, train_fnames, args.model, max_tokens, max_src_tokens, args.sample_mode)
    val_data = APPSBaseDataset(args.train_path, val_fnames, args.model, max_tokens, max_src_tokens, args.sample_mode)
    test_data = APPSBaseDataset(args.train_path, test_fnames, args.model, max_tokens, max_src_tokens, args.sample_mode)

    return train_data, val_data, test_data


def run_training(args, train_data, val_data):
    model_path = args.model_path if args.model_path else f"Salesforce/{args.model}"
    logger.info(f"Loading model from {model_path}")

    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        tuning_mode=args.tuning_mode,
        clone_pl_head=args.clone_pl_head
    )

    if args.clone_pl_head:
        logger.info("Cloning plan head from LM head...")
        lm_head_params = model.lm_head.weight.detach().clone()
        model.pl_head.weight = torch.nn.Parameter(lm_head_params)

    model = model.cuda()
    logger.info(f"Model moved to CUDA. Total GPUs: {torch.cuda.device_count()}")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="epoch",
        
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=500,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=args.save_total_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,

        fp16=args.fp16,
        deepspeed=args.deepspeed
    )

    trainer = Trainer_Plan(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tuning_mode=args.tuning_mode,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_checkpoint")
    model.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


def main(args):
    logger.info("Arguments:")
    logger.info(pprint.pformat(vars(args)))

    os.makedirs(args.save_dir, exist_ok=True)

    train_data, val_data, test_data = get_dataset(args)

    # Save args
    args_path = os.path.join(args.save_dir, "args.json")
    json.dump(vars(args), open(args_path, "w"), indent=4)
    logger.info(f"Saved args to {args_path}")

    run_training(args, train_data, val_data)


if __name__ == "__main__":
    from configs.train_codet5_configs import *  # Assumes `args` is defined there
    main(args)
