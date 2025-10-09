import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

SRC = Path("data/corpus_mlm.csv")
OUT = Path("data/splits/subjects_split.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)
subs = df["subject_id"].dropna().astype(int).unique()

# SPLITS: TRAIN 96,04% | VAL 1.96% | TEST 2%
train_subs, test_subs = train_test_split(subs, test_size=0.02, random_state=42)
train_subs, val_subs  = train_test_split(train_subs, test_size=0.02, random_state=42)

splits = {
    "train": list(map(int, train_subs)),
    "val":   list(map(int, val_subs)),
    "test":  list(map(int, test_subs)),
}

with open(OUT, "w") as f:
    json.dump(splits, f)
print("Salvo:", OUT)