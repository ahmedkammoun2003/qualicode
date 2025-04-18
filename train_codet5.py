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

# On prend des morceaux de code qu’on a faits nous-mêmes
from Datasets_codeT5.apps_dataset import APPSBaseDataset
from trainers.trainer_plan import Trainer_Plan

# On dit à PyTorch comment partager les données quand on utilise plusieurs processus
torch.multiprocessing.set_sharing_strategy('file_system')

# On dit qu’on veut utiliser le GPU numéro 1 pour aller plus vite
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def get_dataset(args):
    """
    📦 Cette fonction va chercher les fichiers dans le dossier du dataset.
    Elle en prend 80% pour apprendre (train) et 20% pour tester (eval).
    """
    fnames = os.listdir(args.train_path)  # On liste tous les fichiers
    if args.db:  # Si on est en mode debug, on ne prend que les 50 premiers
        fnames = fnames[:50]

    # On calcule où couper la liste des fichiers
    split_index = int(0.8 * len(fnames))
    train_fnames, eval_fnames = fnames[:split_index], fnames[split_index:]

    # Si c’est un petit modèle, on change la taille des textes (tokens)
    max_tokens, max_src_tokens = (512, 600) if args.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] else (1024, -1)

    # On fabrique deux datasets : un pour apprendre, un pour évaluer
    train_data = APPSBaseDataset(args.train_path, train_fnames, args.model, max_tokens, args.sample_mode, max_src_tokens)
    eval_data = APPSBaseDataset(args.train_path, eval_fnames, args.model, max_tokens, args.sample_mode, max_src_tokens)

    return train_data, eval_data


def load_model(args):
    """
    🤖 On prend un modèle déjà entraîné (pré-entraîné) qu’on va améliorer.
    S’il faut, on lui ajoute une tête spéciale pour les plans (pl_head).
    """
    model_path = args.model_path if args.model_path else f'Salesforce/{args.model}'
    print(f"Loading model from {model_path}...")

    # On charge le modèle depuis HuggingFace ou un chemin local
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)

    # Si on veut ajouter une tête spéciale pour les plans, on copie lm_head
    if args.clone_pl_head:
        print("Initializing Plan head with finetuned LM head...")
        model.pl_head = torch.nn.Linear(in_features=model.lm_head.in_features,
                                        out_features=model.lm_head.out_features,
                                        bias=False)
        model.pl_head.weight = torch.nn.Parameter(torch.tensor(model.lm_head.weight.detach().numpy()))

    print(f'Finished loading model {args.model}')
    return model


def evaluate_with_codebleu(model, eval_data, args):
    """
    📏 Après l’entraînement, on veut voir si notre modèle est bon.
    On utilise un outil spécial qui s'appelle CodeBLEU pour ça.
    """
    predictions, references = [], []

    # On fait des prédictions pour chaque exemple
    for example in eval_data:
        input_text, reference_text = example['source'], example['target']

        # On transforme le texte en tokens et on l'envoie au modèle
        inputs = args.tokenizer(input_text, return_tensors="pt").to(args.device)
        output_tokens = model.generate(**inputs, max_length=512)

        # On décode la réponse du modèle
        generated_text = args.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        predictions.append(generated_text)
        references.append(reference_text)

    # On calcule un score CodeBLEU (plus c’est haut, mieux c’est)
    score = calc_codebleu(predictions, [references], lang="python")
    print(f"CodeBLEU Score: {score}")
    return score


def run_training(args, train_data, eval_data):
    """
    🏋️‍♂️ Ici on entraîne le modèle avec nos données.
    Ensuite, on regarde s’il a bien appris avec CodeBLEU.
    """
    model = load_model(args)  # On charge le modèle

    # On donne plein d'infos sur comment entraîner le modèle
    training_args = TrainingArguments(
        output_dir=args.save_dir, overwrite_output_dir=True, do_train=True, do_eval=True,
        evaluation_strategy='steps', eval_steps=1, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=5e-6, weight_decay=0.05, lr_scheduler_type='constant_with_warmup',
        logging_dir=args.save_dir, logging_first_step=True, logging_steps=1,
        save_steps=1, save_total_limit=1, dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8, local_rank=args.local_rank,
    )

    # On crée notre entraîneur personnalisé
    trainer = Trainer_Plan(model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data)

    print("Starting training...")  # C’est parti ! Le modèle commence à apprendre
    trainer.train()

    # Si on est le boss (rank 0), on sauvegarde tout et on évalue
    if args.local_rank == 0:
        print("Saving final model checkpoint...")
        trainer.save_model()  # On enregistre le modèle

        # On enregistre les poids dans un fichier .pkl
        torch.save(model.state_dict(), os.path.join(args.save_dir, "final_checkpoint.pkl"))

        # On évalue la qualité du modèle avec CodeBLEU
        score = evaluate_with_codebleu(model, eval_data, args)

        # Et on enregistre ce score dans un fichier JSON
        with open(os.path.join(args.save_dir, "codebleu_score.json"), 'w') as f:
            json.dump({"CodeBLEU": score}, f)


def main(args):
    """
    🚀 C’est la fonction principale qui lance tout le processus.
    Elle prépare les données, entraîne le modèle, puis l’évalue.
    """
    print(pprint.pformat(vars(args)))  # On affiche les paramètres qu’on utilise

    # On crée le dossier où on va tout sauvegarder
    os.makedirs(args.save_dir, exist_ok=True)

    # On enregistre les arguments dans un fichier args.json
    with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    # On récupère les données
    train_data, eval_data = get_dataset(args)

    # On lance l’entraînement
    run_training(args, train_data, eval_data)


# 🧁 Si on lance ce fichier tout seul (et pas importé), alors on exécute main()
if __name__ == "__main__":
    from configs.train_codet5_configs import args
    main(args)
