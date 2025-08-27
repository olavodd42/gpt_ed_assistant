import pickle, os
import pandas as pd
from transformers import AutoTokenizer
from src.data_prep.linearization import build_text_example, _ensure_group_list
IN_PROC = "/home/olavo-dalberto/gpt_ed_assistant/data/processed"

pkl = "/home/olavo-dalberto/gpt_ed_assistant/data/text/train__outcome_critical.pkl"
cpkt = "microsoft/BioGPT"
tok = AutoTokenizer.from_pretrained(cpkt)
if tok.eos_token is None:
    tok.add_special_tokens({"eos_token": "</s>"})

df_train = pd.read_csv(os.path.join(IN_PROC, "train.csv"), low_memory=False)
# df_val = pd.read_csv(os.path.join(IN_PROC, "valid.csv"), low_memory=False)
# df_test = pd.read_csv(os.path.join(IN_PROC, "test.csv"), low_memory=False)
if "lab_group_idx" in df_train.columns:
    df_train["lab_group_idx"] = df_train["lab_group_idx"].apply(_ensure_group_list)

# with open(pkl, "rb") as f:
#     data = pickle.load(f)

# ex = data[0]
# txt = tok.decode(ex["input_ids"], skip_special_tokens=False)
# print(txt[:400])
# print("EOS idx:", [i for i,t in enumerate(ex["input_ids"]) if t == tok.eos_token_id])

row = df_train.iloc[0]
txt = build_text_example(row, row["lab_group_idx"], tok.eos_token)
print(txt)
print("qtd <eos>:", txt.count(tok.eos_token))

print("lab_groups:", row["lab_group_idx"], type(row["lab_group_idx"]))
