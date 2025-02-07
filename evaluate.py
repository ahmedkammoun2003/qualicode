import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu import corpus_bleu
from datasets import load_dataset
import os
from codebleu import calc_codebleu  # Import de CodeBLEU

# Charger le modèle fine-tuné
model_path = "./models/final_checkpoint.pkl"
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-large")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")

# Charger les poids sauvegardés
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def generate_code(prompt, max_length=100, temperature=0.7):
    """Génère du code à partir d'un prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=max_length, temperature=temperature, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def evaluate_codebleu(reference, hypothesis, lang="python"):
    """Calcule le score CodeBLEU entre un code généré et un code de référence."""
    return calc_codebleu([reference], [hypothesis], lang)

def test_code_execution(code):
    """Teste si le code généré s'exécute sans erreur."""
    try:
        exec(code, {})
        return "Succès"
    except Exception as e:
        return f"Erreur: {e}"

# Tester avec un exemple
prompt = "def fibonacci(n):"
reference_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""".strip()

generated_code = generate_code(prompt)
codebleu_score = evaluate_codebleu(reference_code, generated_code)
execution_result = test_code_execution(generated_code)

print("=== Code généré ===")
print(generated_code)
print("\n=== Évaluation ===")
print(f"CodeBLEU Score: {codebleu_score}")
print(f"Exécution: {execution_result}")

# Évaluation sur MBPP
def evaluate_on_mbpp():
    dataset = load_dataset("google-research-datasets/mbpp", "full")
    scores = []
    i = 0
    for sample in dataset["test"]:  
        i += 1
        sample = dict(sample)
        prompt = sample["text"]
        print(f"Test {i} : {prompt}")
        reference = sample["code"].strip()
        generated = generate_code(prompt)
        print(generated)
        score = evaluate_codebleu(reference, generated)
        scores.append(score)  # Ajouter tous les scores CodeBLEU
    
    # Calculer la moyenne de chaque métrique CodeBLEU
    avg_scores = {key: sum(d[key] for d in scores) / len(scores) for key in scores[0]}
    print(f"Scores moyens CodeBLEU sur MBPP: {avg_scores}")

evaluate_on_mbpp()