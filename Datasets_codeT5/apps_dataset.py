# Imports nécessaires pour la gestion des données, modèles et utilitaires
import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 
import sys

from multiprocessing import Manager
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import Datasets_codeT5.utils as dsutils

torch.cuda.empty_cache()  # Vide le cache GPU pour éviter les surcharges mémoire

# Classe de dataset personnalisé pour le benchmark APPS, utilisable dans un DataLoader PyTorch
class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode, max_src_tokens):
        self.dataroot = dataroot  # Répertoire racine contenant les données
        self.problem_dirs = problem_dirs  # Liste des dossiers de problèmes (chaque problème = 1 tâche)

        self.model = model  # Nom du modèle (ex: codet5)
        self.sample_mode = sample_mode  # Méthode d'échantillonnage (non utilisé dans ce code sauf pour vérifier)

        self.max_tokens = max_tokens  # Longueur max de la séquence
        self.max_src_tokens = max_src_tokens  # Longueur max pour l'entrée uniquement

        self.samples = []  # Liste des échantillons GT (solutions finales)
        self.all_error_types, self.all_error_subtypes, self.all_baseline_error_types = [], [], []
        self.initialize()  # Charge les données

        # Initialisation du tokenizer
        if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    # Fonction pour charger les plans de raisonnement
    def load_plan_samples(self, sols_plan, answer_type, starter_code, question_str): 
        samples = []
        for idx, plan_str in enumerate(sols_plan):   
            sample = (question_str, starter_code, plan_str, answer_type)
            samples.append(sample)
        return samples
     
    # Fonction pour charger les solutions finales (ground truth)
    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)  # Réindentation automatique
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)
        return samples

    # Fonction principale de chargement de tous les fichiers des problèmes
    def initialize(self):
        all_samples = []
        plan_samples = []
        skipped_problems = []

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):           
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")     
            plans_fname = os.path.join(self.dataroot, problem_name, 'plans.json')       

            # Vérifie que les fichiers nécessaires existent
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue
            if not os.path.exists(plans_fname):
                plans_fname = None

            # Lecture de la question
            with open(question_fname, 'r', encoding='utf-8') as f:
                question_str = f.read()
            
            # Lecture du code de démarrage si présent
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")    
            if os.path.isfile(starter_code):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            # Chargement des solutions
            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)

            # Chargement des plans si disponible
            if plans_fname:
                with open(plans_fname, 'r', encoding='utf-8') as f:
                    sols_plan_list = json.load(f)
                plan_sample = self.load_plan_samples(sols_plan_list, answer_type, starter_code, question_str)
                plan_samples += plan_sample

            all_samples += gt_samples
                
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        
        self.plan_samples = plan_samples
        self.samples = all_samples

    # Longueur du dataset
    def __len__(self):
        return len(self.samples)

    # Accès à un élément individuel : assemble la tâche et son plan
    def __getitem__(self, idx):
        raw_samples = self.pack_samples(idx)
        inputs = self.sample_task(raw_samples)

        plan_sample_idx = random.randint(0, len(self.plan_samples)-1)
        plan_samples = self.pack_samples(plan_sample_idx, "plan")

        plan_inputs = self.sample_plan_task(plan_samples)

        # Ajoute les plans en tant que champs supplémentaires dans l'entrée
        for k, v in plan_inputs.items():
            inputs['pl_{}'.format(k)] = v

        gc.collect()  # Nettoie la mémoire GPU

        return inputs

    # Emballe les échantillons jusqu’à atteindre un seuil de tokens
    def pack_samples(self, idx, sample_type=None):
        curr_num_tokens = 0
        curr_samples = [] 
        sample_pool = self.plan_samples if sample_type == 'plan' else self.samples
        if len(sample_pool) == 0:
            raise ValueError("Sample pool is empty")
        
        curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]

        while curr_num_tokens < self.max_tokens:
            # Coupe si trop long
            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            # Comptage des tokens pour chaque composante
            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))            
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))
            
            # On ne prend qu’un échantillon à la fois pour CodeT5
            if self.model in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"]:
                break 

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool)-1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx] 
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    # Prépare l'entrée pour la génération de code
    def sample_task(self, samples, sample_type=None):
        input_ids = []
        label_ids = []

        for sample in samples:
            q_str, s_str, a_str, answer_type = sample
            
            q_str =  "[GEN_CODE]"+"\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"
            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            input_ids.extend(question_token_ids)
             
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            if self.model not in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"]:
                label_ids.extend([-100] * len(question_token_ids))
                answer_token_ids.append(self.tokenizer.eos_token_id)
                input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)

        # Padding et coupe à la taille max
        input_ids_max_len = int(self.max_src_tokens) if self.model in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"] else int(self.max_tokens)
        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 

            if self.model not in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"]:
                new_label_ids = [-100] * input_ids_max_len 
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids
                
        if self.model in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens 
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids
        
        if self.model not in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"] and len(input_ids) != len(label_ids):
            pdb.set_trace()
        
        input_ids = input_ids[:input_ids_max_len]
        label_ids = label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            "labels" :  torch.LongTensor(label_ids)
        }           
        return out_sample 

    # Prépare l’entrée pour la génération de plan (raisonnement)
    def sample_plan_task(self, plan_samples):
        input_ids = []
        label_ids = []
                    
        for sample in plan_samples:
            q_str, s_str, p_str, answer_type = sample
            
            q_str =  "[GEN_PLAN]"+"\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nLet's think step by step:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            input_ids.extend(question_token_ids)
             
            answer_token_ids = self.tokenizer.encode(p_str, verbose=False)
            if self.model not in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"]:
                label_ids.extend([-100] * len(question_token_ids))
                answer_token_ids.append(self.tokenizer.eos_token_id)
                input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)

        # Même logique de padding que sample_task
        input_ids_max_len = int(self.max_src_tokens) if self.model in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"] else int(self.max_tokens)
        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 
            
            if self.model not in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"]:
                new_label_ids = [-100] * input_ids_max_len 
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids
                
        if self.model in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens 
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids
        
        if self.model not in ['codet5-base', 'codet5-large', "codet5-large-ntp-py"] and len(input_ids) != len(label_ids):
            pdb.set_trace()
        
        input_ids = input_ids[:input_ids_max_len]
        label_ids = label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            'plan_id': [1],
            "labels" :  torch.LongTensor(label_ids)
        }           
        return out_sample
