import os
import re
import json

base_path = "."  # ← remplace par le bon chemin

for i in range(5000):
    num = f"{i}"
    txt_file = os.path.join(base_path, f"{num}_plans.txt")
    output_folder = os.path.join(base_path, f"{i:04d}")
    output_file = os.path.join(output_folder, "plans.json")  # ← nom du JSON

    if not os.path.isfile(txt_file):
        continue

    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Supprimer tout ce qui est entre <think> et </think>
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Nettoyer les espaces et retours à la ligne
    cleaned = " ".join(cleaned.split())

    # S'assurer que le dossier existe
    os.makedirs(output_folder, exist_ok=True)

    # Sauvegarder directement le texte (pas un objet JSON)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)

    print(f"✅ {txt_file} → {output_file} (fichier texte supprimé)")
