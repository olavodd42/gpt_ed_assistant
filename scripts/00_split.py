import os
import pandas as pd
from sklearn.model_selection import train_test_split

OUT_PATH = "./data/interim"
df_train = pd.read_csv("./data/raw/processed/train.csv")
df_test  = pd.read_csv("./data/raw/processed/test.csv")

# TRAIN: 64%, VALID: 16%, TEST: 20%
df_train_final, df_valid = train_test_split(
    df_train,
    test_size=0.2,     # 20% do conjunto de treino vira validação
    stratify=df_train["outcome_hospitalization"],  # manter balanceamento do target
    random_state=42    # reprodutibilidade
)

try:
    df_train_final.to_csv(os.path.join(OUT_PATH, "train.csv"))
    df_valid.to_csv(os.path.join(OUT_PATH, "valid.csv"))
    df_test.to_csv(os.path.join(OUT_PATH, "test.csv"))
    print(f"DATA SAVED TO {OUT_PATH}")
except Exception as e:
    print(f"Unexpected error: {e}")