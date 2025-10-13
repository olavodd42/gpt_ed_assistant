#!/usr/bin/env python3
import pandas as pd, json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Caminhos
CSV_PATH = "data/master_dataset.csv"             # ajuste o nome se estiver diferente
OUT_PATH = Path("data/splits/subjects_split.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1️⃣ Carrega o CSV (só precisamos de subject_id)
df = pd.read_csv(CSV_PATH, usecols=["subject_id"])
print("Registros totais:", len(df))

# 2️⃣ Lista única de pacientes
subs = df["subject_id"].dropna().unique().astype(int)
print("Pacientes únicos:", len(subs))

# 3️⃣ Divide 80%/10%/10% por paciente (não por linha)
train_subs, temp = train_test_split(subs, test_size=0.2, random_state=42)
val_subs, test_subs = train_test_split(temp, test_size=0.5, random_state=42)

splits = {
    "train": train_subs.tolist(),
    "val":   val_subs.tolist(),
    "test":  test_subs.tolist(),
}

# 4️⃣ Salva
with open(OUT_PATH, "w") as f:
    json.dump(splits, f, indent=2)

print(f"✅ Split salvo em {OUT_PATH}")
print(f"Train: {len(train_subs)} | Val: {len(val_subs)} | Test: {len(test_subs)}")
