import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from configs.generate_codet5_configs import * 

def main():
    # Chargement du tokenizer et du modèle
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    
    # Set up device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize base model first
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    
    # Add pl_head layer with same dimensions as lm_head
    model.pl_head = torch.nn.Linear(in_features=model.lm_head.in_features,
                                   out_features=model.lm_head.out_features,
                                   bias=False)
    
    # Load the trained weights from pkl file
    pkl_path = './models/final_checkpoint.pkl'
    print(f"Loading model from {pkl_path}")
    state_dict = torch.load(pkl_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    print("\n=== Générateur de code avec CodeT5 ===\n")

    is_plan = input("Souhaitez-vous générer un plan ? (y/n) : ").strip().lower() == 'y'

    # Récupération de l'énoncé
    question = input("\nEntrez l'énoncé du problème (multi-lignes, finissez avec une ligne vide) :\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "": 
            break
        lines.append(line)
    question = "\n".join(lines)

    # Starter code optionnel
    add_starter = input("\nSouhaitez-vous ajouter un code de départ ? (y/n) : ").strip().lower() == 'y'
    starter_code = ""
    if add_starter:
        print("Entrez le starter code (finissez avec une ligne vide) :")
        starter_lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            starter_lines.append(line)
        starter_code = "\n" + "\n".join(starter_lines)

    # Création de l'input du modèle
    prompt_type = "[GEN_PLAN]" if is_plan else "[GEN_CODE]"
    input_text = f"{prompt_type}\nQUESTION:\n{question}"
    input_text += "\nLet's think step by step:\n" if is_plan else "\nANSWER:\n"

    # Get generation parameters first
    num_outputs = int(input("\nCombien de solutions voulez-vous générer ? (ex: 3) : "))
    temperature = float(input("Température pour la génération (ex: 0.7) : "))

    print("\n⏳ Génération en cours...\n")
    
    # First generate the code
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_length=512,
        num_return_sequences=num_outputs,
        top_p=0.95
    )

    # Print the generated code and generate tests
    for i, output in enumerate(output_ids):
        code = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n=== Solution {i+1} ===\n{code}")
        
        # Generate tests for the code with a more specific prompt
        test_prompt = f"{prompt_type}\nQUESTION:\nWrite comprehensive unit tests using Python's unittest framework for this code. Include test cases for normal input, edge cases, and error cases:\n{code}\nANSWER:\n"
        test_input_ids = tokenizer.encode(test_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        test_output_ids = model.generate(
            test_input_ids,
            do_sample=True,
            temperature=min(temperature, 0.7),  # Lower temperature for more focused test generation
            max_length=512,
            num_return_sequences=1,
            top_p=0.95
        )
        test_code = tokenizer.decode(test_output_ids[0], skip_special_tokens=True)
        print(f"\n=== Tests for Solution {i+1} ===\n{test_code}\n")

    num_outputs = int(input("\nCombien de solutions voulez-vous générer ? (ex: 3) : "))
    temperature = float(input("Température pour la génération (ex: 0.7) : "))

    print("\n⏳ Génération en cours...\n")
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_length=512,
        num_return_sequences=num_outputs,
        top_p=0.95
    )

    for i, output in enumerate(output_ids):
        code = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n=== Solution {i+1} ===\n{code}")

if __name__ == "__main__":
    main()
