import pandas as pd, json

CSV_PATH = "data/master_dataset.csv"
SPLIT_JSON = "data/splits/subjects_split.json"
OUT_PATH = "data/master_dataset_split.parquet"

# Carrega dados
df = pd.read_csv(CSV_PATH)
with open(SPLIT_JSON) as f: split = json.load(f)

def get_split(sid):
    sid = int(sid)
    if sid in split["train"]: return "train"
    elif sid in split["val"]: return "val"
    elif sid in split["test"]: return "test"
    else: return "ignore"

df["split"] = df["subject_id"].apply(get_split)
print(df["split"].value_counts())

# Salva
df.to_parquet(OUT_PATH, index=False)
print(f"✅ Dataset salvo com coluna split: {OUT_PATH}")
