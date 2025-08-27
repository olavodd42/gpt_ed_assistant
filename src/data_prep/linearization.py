import pandas as pd
import json
from typing import Dict, List
from ast import literal_eval
from src.utils.seed import seed

seed()

with open('/home/olavo-dalberto/gpt_ed_assistant/data/processed/feature_map.json') as f:
    FEATURE_MAP = json.load(f)  # ex.: {"feature1": "age", ...}

ORIG2FEAT: Dict[str, str] = {orig: feat for feat, orig in FEATURE_MAP.items()}


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
TARGET_COLS = ["outcome_critical", "outcome_ed_los", "lab_group_idx"]


def _ensure_group_list(x):
    """
    Converte lab_groups em List[int].
    Aceita: list, tuple, numpy array, string tipo "[0, 1, 5]".
    Filtra ids fora de 0..11.
    """
    if x is None:
        return []
    # já é lista/tupla -> vira lista
    if isinstance(x, (list, tuple)):
        lst = list(x)
    # string -> tenta literal_eval
    elif isinstance(x, str):
        try:
            lst = literal_eval(x)
        except Exception:
            # fallback: separa por vírgula e mantém dígitos
            lst = [int(t.strip()) for t in x.strip("[](){}").split(",") if t.strip().isdigit()]
    else:
        # tenta embrulhar único valor
        try:
            lst = list(x)
        except Exception:
            lst = [int(x)]
    # garante ints e faixa válida
    out = []
    for v in lst:
        try:
            iv = int(v)
            if 0 <= iv <= 11:
                out.append(iv)
        except Exception:
            pass
    return out


def _val_from_original(row: pd.Series, original_name: str):
    """Busca o valor no row usando o nome original (mapeando para featureX)."""
    feat = ORIG2FEAT.get(original_name)
    if feat is None or feat not in row:
        return None
    return row[feat]

def linearize_ehr(row: pd.Series) -> str:
    """
    Converte os dados tabulares em texto legível para o tokenizer compreender.
    Parâmetros:
        - row: pd.Series
    Retorna:
        - string
    """
    parts = []
    for orig in ED_EHR:
        v = _val_from_original(row, orig)

        if pd.isna(v) or v is None:
            continue

        parts.append(f"{orig}: {v}")
    
    return '; '.join(parts)

def linearize_group(row: pd.Series, group_idx: int) -> str:
    """
    Converte os dados dos grupos laboratoriais em formato de string.
    Parâmetros:
        - row: pd.Series
        - group_idx: int
    Retorna:
        - string
    """

    cols = ED_LAB_IDX.get(group_idx, [])
    vals = []
    for orig in cols:
        v = _val_from_original(row, orig)
        if v is None or pd.isna(v):
            continue

        vals.append(f"{orig}: {v}")

    if not vals:
        return ""
    
    gname = GROUP_NAME.get(group_idx, f"group_{group_idx}")
    return f"{gname}: " + " | ".join(vals)

def build_text_example(row, lab_groups, eos_token: str) -> str:
    """
    Concatena triagem e grupos de labs com <eos> entre blocos.
    Ex.: "Age: 67; Gender: M; ... <eos> cbc: hemoglobin: 13.2 | ... <eos> ... <eos>"
    """
    parts = []

    # 1) bloco de triagem 
    ehr_txt = linearize_ehr(row)          # <- STRING
    if ehr_txt:
        parts.append(ehr_txt)
    parts.append(eos_token)                # <eos> depois da triagem

    # g = 0  # ex: CBC
    # print([ (c, row.get(ORIG2FEAT.get(c,''), None)) for c in ED_LAB_IDX[g] ])
    groups = _ensure_group_list(lab_groups)

    # 2) blocos de laboratório (1 por grupo presente)
    for g in lab_groups:
        gtxt = linearize_group(row, g)
        if not gtxt:
            continue
        parts.append(gtxt)
        parts.append(eos_token)            # <eos> separando os grupos

    # 3) <eos> final (rótulo)
    parts.append(eos_token)

    return " ".join(parts)



def save_to_txt(txt: str, filename: str) -> None:
    """
    Salva um texto em um arquivo .txt.
    Parâmetros:
        - txt: str
        - filename: str
    """
    with open(f".../data/text/{filename}", 'w') as f:
        f.write(txt)