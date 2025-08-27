import os, sys
import pandas as pd
from ast import literal_eval
from src.utils.seed import seed
from src.data_prep.linearization import build_text_example, save_to_txt
from src.data_prep.tokenizer import encode_split_per_target
from src.utils.tokenization import load_tokenizer, save_tokenizer

seed()
IN_PROC = "/home/olavo-dalberto/gpt_ed_assistant/data/processed"
BASE_CPKT = "microsoft/BioGPT"
TARGETS = ["outcome_critical", "outcome_ed_los", 'lab_group_idx']

df_train = pd.read_csv(os.path.join(IN_PROC, "train.csv"), low_memory=False)
df_val = pd.read_csv(os.path.join(IN_PROC, "valid.csv"), low_memory=False)
df_test = pd.read_csv(os.path.join(IN_PROC, "test.csv"), low_memory=False)

# verifica se lab_group_idx está como string, se for converte lista
for df in (df_train, df_val, df_test):
    if df["lab_group_idx"].dtype == object:
        df["lab_group_idx"] = df["lab_group_idx"].apply(
            lambda x: x if isinstance(x, list) else literal_eval(str(x))
        )

tokenizer = load_tokenizer(BASE_CPKT)

if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "<eos>"})

meta_extra = {
    "ehr_cols": [
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
    ],         # se quiser, injete seus nomes (mapa/labels)
    "ed_lab_idx": {
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
    },
    "group_name": {
        0: "cbc", 1: "chem", 2: "coag", 3: "ua", 4: "lactate",
        5: "lfts", 6: "lipase", 7: "lytes", 8: "cardio",
        9: "blood gas", 10: "tox", 11: "inflammation"
    }
}


encode_split_per_target(df_train, "train", tokenizer.eos_token, tokenizer, TARGETS, build_text_example, meta_extra)
encode_split_per_target(df_val, "valid", tokenizer.eos_token, tokenizer, TARGETS, build_text_example, meta_extra)
encode_split_per_target(df_test,  "test",  tokenizer.eos_token, tokenizer, TARGETS, build_text_example, meta_extra)
save_tokenizer(tokenizer)