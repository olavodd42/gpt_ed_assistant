import pandas as pd
import json
from typing import Dict, List
from src.utils.seed import seed

seed()

with open('/home/olavo-dalberto/gpt_ed_assistant/data/processed/feature_map.json') as f:
    ED_EHR_MAP = json.load(f)

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
TARGET_COLS = ["outcome_critical", "outcome_ed_los"]

def linearize_ehr(row: pd.Series) -> str:
    """
    Converte os dados tabulares em texto legível para o tokenizer compreender.
    Parâmetros:
        - row: pd.Series
    Retorna:
        - string
    """
    parts = []
    for col in ED_EHR:
        if col not in row:
            continue

        label = ED_EHR_MAP.get(col, col)
        val = row[col]
        parts.append(f"{label}: {val}")
    
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
    if not cols:
        return ""
    pairs = []
    for c in cols:
        if c in row:
            pairs.append(f"{c}: {row[c]}")
    if not pairs:
        return ""
    
    gname = GROUP_NAME.get(group_idx, f"group_{group_idx}")
    return f"{gname}: " + " | ".join(pairs)

def build_text_example(row: pd.Series, lab_groups: List[int], eos_token: str) -> str:
    """
    Pega uma linha tabular e transforma em texto estruturado + <eos> delimitadores."
    Parâmetros:
        - row: pd.Series
        - lab_groups: List[int]
        - eos_token: str
    Retorna:
        - str
    """
    ehr_txt = linearize_ehr(row)
    chunks = [ehr_txt + f" {eos_token}"]

    for gi in lab_groups:
        gtxt = linearize_group(row, gi)
        if gtxt:
            chunks.append(gtxt + f" {eos_token}")

    return " ".join(chunks).strip()

def save_to_txt(txt: str, filename: str) -> None:
    """
    Salva um texto em um arquivo .txt.
    Parâmetros:
        - txt: str
        - filename: str
    """
    with open(f".../data/text/{filename}") as f:
        f.write(txt)