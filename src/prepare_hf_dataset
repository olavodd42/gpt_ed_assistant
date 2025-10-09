#!/usr/bin/env python3
import os, json
from typing import Dict, Set
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# ====== Parâmetros ======
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN    = 256
STRIDE     = 64
CSV_PATH   = "data/corpus_mlm.csv"
SPLIT_JSON = "data/splits/subjects_split.json"
OUT_DIR    = "data/ds_mlm_bioclinicalbert"
BATCH_SIZE = 1000        
NUM_PROC   = 1

# Dica: evita manter tudo em memória nos pipelines do HF
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")

# ====== Utils ======
def load_split_sets(path: str) -> Dict[str, Set[str]]:
    with open(path) as f:
        sp = json.load(f)
    # Converta para str para comparação direta (sem cast por linha)
    return {k: set(map(str, v)) for k, v in sp.items()}

def filter_by_subject(split_set: Set[str]):
    # Função de filtro: aceita str ou num (convertemos só se precisar)
    def _f(ex):
        sid = ex["subject_id"]
        # alguns CSVs salvam como int; garanta string
        if not isinstance(sid, str):
            sid = str(sid)
        return sid in split_set
    return _f

def tokenize_batch(tokenizer: AutoTokenizer):
    # Tokeniza em lotes, gerando janelas com stride automaticamente
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_attention_mask=False,
            # atenção: o collator de MLM cuida do masking, então aqui só ids
        )
    return _tok

def prepare_one_split(split_name: str, subj_set: Set[str], tokenizer: AutoTokenizer):
    # 1) Carrega CSV como Dataset (não usa pandas)
    ds = load_dataset("csv", data_files=CSV_PATH, split="train")
    # 2) Filtra por subject_id do split (stream de colunas leves)
    ds = ds.filter(filter_by_subject(subj_set), num_proc=NUM_PROC)

    # 3) Opcional: mantenha só colunas que você quer para debug/traço
    keep_cols = ["subject_id","hadm_id","note_id","category","text"]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    # 4) Tokeniza em lote com overflow (gera as janelas já “explodidas”)
    ds_tok = ds.map(
        tokenize_batch(tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=keep_cols,   # remove texto cru
        desc=f"Tokenizando {split_name}",
    )

    # ds_tok conterá 'input_ids' (e 'overflow_to_sample_mapping' é descartado automaticamente)
    return ds_tok

def main():
    print(">>> Carregando splits...")
    splits = load_split_sets(SPLIT_JSON)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print(">>> Preparando train...")
    ds_train = prepare_one_split("train", splits["train"], tok)
    ds_train.save_to_disk(f"{OUT_DIR}_train")

    print(">>> Preparando validation...")
    ds_val = prepare_one_split("validation", splits["val"], tok)
    ds_val.save_to_disk(f"{OUT_DIR}_val")

    print(">>> Preparando test...")
    ds_test = prepare_one_split("test", splits["test"], tok)
    ds_test.save_to_disk(f"{OUT_DIR}_test")

    # (Opcional) Empacotar em um DatasetDict leve (aponta para discos separados)
    ds = DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})
    ds.save_to_disk(OUT_DIR)
    print("✅ Salvo em", OUT_DIR)
    print(ds)
    
if __name__ == "__main__":
    main()
