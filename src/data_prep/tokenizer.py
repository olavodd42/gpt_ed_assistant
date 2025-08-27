import pickle
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase, BatchEncoding
from src.utils.seed import seed 

seed()
with open('/home/olavo-dalberto/gpt_ed_assistant/data/processed/feature_map.json') as f:
    EHR_LABEL_MAP = json.load(f)

ED_EHR = [
    "age", "gender", 
            
    "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
    "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d",
    
    "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
    "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
    "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
    "cci_Cancer2", "cci_HIV",  

    "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
    "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
    "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
    "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression",
    
    #VS
    "triage_temperature",
    "triage_heartrate",
    "triage_resprate", 
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    #RNHX
    "triage_pain",
    "triage_acuity",
    "chiefcomplaint",
]
ED_LAB_IDX: Dict[int, List[str]] = {
    0: [#CBC
    'hematocrit',
    'white blood cells',
    'hemoglobin',
    'red blood cells',
    'mean corpuscular volume',
    'mean corpuscular hemoglobin',
    'mean corpuscular hemoglobin concentration',
    'red blood cell distribution width',
    'platelet count',
    'basophils',
    'eosinophils',
    'lymphocytes',
    'monocytes',
    'neutrophils',
    'red cell distribution width (standard deviation)',
    'absolute lymphocyte count',
    'absolute basophil count',
    'absolute eosinophil count',
    'absolute monocyte count',
    'absolute neutrophil count',
    'bands',
    'atypical lymphocytes',
    'nucleated red cells'],
    1: [#CHEM
    'urea nitrogen',
    'creatinine',
    'sodium',
    'chloride',
    'bicarbonate',
    'glucose (chemistry)',
    'potassium',
    'anion gap',
    'calcium, total'],
    2: [
    'prothrombin time',
    'inr(pt)',
    'ptt'
    ],
    3: [
    'ph (urine)',
    'specific gravity',
    'red blood count (urine)',
    'white blood count (urine)',
    'epithelial cells',
    'protein',
    'hyaline casts',
    'ketone',
    'urobilinogen',
    'glucose (urine)'
    ],
    4: [
    'lactate'
    ],
    5: [#LFTs
    'alkaline phosphatase',
    'asparate aminotransferase (ast)',
    'alanine aminotransferase (alt)',
    'bilirubin, total',
    'albumin'],
    6: [#LIPASE
    'lipase',],
    7: [#LYTES
    'magnesium',
    'phosphate'],
    8: [#CARDIO
    'ntprobnp',
    'troponin t'],
    9: [#BLOOD_GAS
    'potassium, whole blood',
    'ph (blood gas)',
    'calculated total co2',
    'base excess',
    'po2',
    'pco2',
    'glucose (blood gas)',
    'sodium, whole blood'],
    10: [#TOX
    'ethanol'],
    11: [#INFLAMMATION
    'creatine kinase (ck)',
    'c-reactive protein']   
}
GROUP_NAME = {
    0: "cbc", 1: "chem", 2: "coag", 3: "ua", 4: "lactate",
    5: "lfts", 6: "lipase", 7: "lytes", 8: "cardio",
    9: "blood gas", 10: "tox", 11: "inflammation"
}
OUT_DIR = "/home/olavo-dalberto/gpt_ed_assistant/data/text"
MAX_LEN = 256
BASE_CPKT = "microsoft/BioGPT"


def _coerce_scalar_label(v) -> int:
    """
    Converte v para rótulo escalar {0,1} de forma robusta.
    Aceita: bool, int/float, str, numpy scalar, e containers 1D (list/ndarray/Series).
    """
    # 1) Desembrulhar containers (list/tuple/ndarray/Series)
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        # se vier um único valor dentro, usa esse; se vier vazio -> 0; se vier >1, pega o primeiro
        if len(v) == 0:
            return 0
        v = v[0]

    # 2) Tratar NaN/None
    # (agora v é escalar; usar pd.isna é seguro)
    if pd.isna(v):
        return 0

    # 3) Tipos numéricos/booleanos
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v != 0)
    if isinstance(v, (float, np.floating)):
        return int(float(v) >= 0.5) if (v >= 0.0 and v <= 1.0) else int(v != 0)

    # 4) Strings
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return 1
        if s in {"0", "false", "f", "no", "n"}:
            return 0
        # tentar numérico em string
        try:
            fv = float(s)
            return int(fv >= 0.5) if (0.0 <= fv <= 1.0) else int(fv != 0)
        except Exception:
            # qualquer outra string vira 0
            return 0

    # 5) Fallback
    try:
        return int(v != 0)
    except Exception:
        return 0

def encode_split_per_target(df: pd.DataFrame, split_name: str, eos: str, tokenizer: PreTrainedTokenizerBase, target_cols: List[str], build_text_example_fn, meta_extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Lineariza os dados em texto e tokeniza ele. Cria um arquivo ppl por alvo.
    Parâmetros:
        - df: pd.Dataframe
        - split_name: str
    """
    if df["lab_group_idx"].dtype != object or not isinstance(df["lab_group_idx"].iloc[0], list):
        df = df.copy()
        df["lab_group_idx"] = df["lab_group_idx"].apply(lambda x: x if isinstance(x, list) else list(x))

    encoded_records = []
    for _,row in df.iterrows():
        # Obtém grupo de laboratório (id) e lineariza texto
        lab_groups = row["lab_group_idx"] if "lab_group_idx" in row else []
        text = build_text_example_fn(row, lab_groups=lab_groups, eos_token=eos)

        # Tokeniza o texto
        encoded_df = tokenizer(
            text,
            truncation=True,
            add_special_tokens=False,
            max_length=MAX_LEN,
            padding="max_length",
            return_attention_mask=True
        )

        # Obtém os valores da variável alvo (y)
        # y = int(row[target_col]) if target_col in row else 0

        # Guarda os elementos em uma lista de dicionários
        rec = {
                "input_ids": encoded_df["input_ids"],
                "attention_mask": encoded_df["attention_mask"],
                "lab_groups": lab_groups,
                "raw_text": text
            }
        encoded_records.append(rec)

    # Armazena o valor de cada variável alvo
    os.makedirs(OUT_DIR, exist_ok=True)
    for tgt in target_cols:
        if tgt not in df.columns:
            raise KeyError(f"Target '{tgt}' não está no DataFrame.")
        
        records = []
        for base, row in zip(encoded_records, df.itertuples(index=False)):
            y = getattr(row, tgt)
            y = _coerce_scalar_label(y)
            rec = dict(base)
            rec["label"] = y
            records.append(rec)
        
        # salva no formato pickle
        out_pkl = os.path.join(OUT_DIR, f"{split_name}__{tgt}.pkl")

        with open(out_pkl, "wb") as f:
            pickle.dump(records, f)

        print(f"[OK] {split_name} ({tgt}): {len(records)} exemplos -> {out_pkl}")

        meta = {
            "base_model_path": BASE_CPKT,
            "max_len": MAX_LEN,
            "target_cols": target_cols,
            "eos_token": eos,
        }
        if meta_extra:
            meta.update(meta_extra)

        with open(os.path.join(OUT_DIR, "linearize_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("[OK] meta -> linearize_meta.json")

        # tokenizer.save_pretrained("/home/olavo-dalberto/gpt_ed_assistant/experiments/models")