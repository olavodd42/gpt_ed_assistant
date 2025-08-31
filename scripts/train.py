import os
import json
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.tok.tokenize import load_tokenizer, tokenize_texts
from src.dataio.reader import read_dataset
from src.dataset.clinical import ClinicalDataset, collate_ed
from src.trainer.ModelTrainer import ModelTrainer
from src.utils.seed import seed
seed()

def cleanup():
    gc.collect()                     # força o garbage collector do Python
    torch.cuda.empty_cache()         # limpa cache de memória que não está em uso
    torch.cuda.ipc_collect()         # limpa memória compartilhada entre processos (às vezes útil)

ed_ehr = [
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

ed_lab = [
    #CBC
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
    'nucleated red cells',
    #CHEM
    'urea nitrogen',
    'creatinine',
    'sodium',
    'chloride',
    'bicarbonate',
    'glucose (chemistry)',
    'potassium',
    'anion gap',
    'calcium, total',
    #COAG
    'prothrombin time', 'inr(pt)', 'ptt',
    #UA
    'ph (urine)',
    'specific gravity',
    'red blood count (urine)',
    'white blood count (urine)',
    'epithelial cells',
    'protein',
    'hyaline casts',
    'ketone',
    'urobilinogen',
    'glucose (urine)',
    #LACTATE
    'lactate',
    #LFTs
    'alkaline phosphatase',
    'asparate aminotransferase (ast)',
    'alanine aminotransferase (alt)',
    'bilirubin, total',
    'albumin',
    #LIPASE
    'lipase',
    #LYTES
    'magnesium', 'phosphate',
    #CARDIO,
    'ntprobnp', 'troponin t',
    #BLOOD_GAS
    'potassium, whole blood',
    'ph (blood gas)',
    'calculated total co2',
    'base excess',
    'po2',
    'pco2',
    'glucose (blood gas)',
    'sodium, whole blood',
    #TOX
    'ethanol',
    #INFLAMMATION
    'creatine kinase (ck)', 'c-reactive protein',
    
]

outcome_cols = [
    'outcome_critical',
    'outcome_ed_los',
    'lab_group_idx'
]

ED_EHR = [f'feature{i+1}' for i in range(len(ed_ehr))]
ED_LAB = [f'feature{i+1}' for i in range(len(ed_ehr), len(ed_lab))]
feature_map = dict(zip(ED_EHR + ED_LAB, ed_ehr + ed_lab))
GROUP_NAME = {
    0: "cbc", 1: "chem", 2: "coag", 3: "ua", 4: "lactate",
    5: "lfts", 6: "lipase", 7: "lytes", 8: "cardio",
    9: "blood gas", 10: "tox", 11: "inflammation"
}

CPKT = "microsoft/BioGPT"
MAX_LEN = 256
BATCH = 1

# 1) ler CSV -> textos + labels + grupos
df_train = pd.read_csv("data/processed/train.csv")
tokenizer = load_tokenizer(CPKT)

texts, y, groups, _ = read_dataset(
    df_train, task_col="outcome_critical",
    eos_token=tokenizer.eos_token,
    ehr_cols=ED_EHR, group_name=GROUP_NAME, group_cols=ED_LAB
)

# 2) tokenizar
enc = tokenize_texts(tokenizer, texts, MAX_LEN)

# 3) dataset/loader
dataset = ClinicalDataset(enc, y, groups)
loader  = DataLoader(
    dataset, batch_size=BATCH, shuffle=True, num_workers=0,
    collate_fn=lambda b: collate_ed(b, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id)
)

trainer = ModelTrainer(
    cpkt=CPKT, tokenizer=tokenizer,
    clf_loss=nn.CrossEntropyLoss(), sel_loss=nn.BCEWithLogitsLoss(),
    baseline=False, use_amp=True, grad_clip=1.0, accumulation_steps=16
)

trainer.configure_optimizer(optim.AdamW, lr=2e-5, weight_decay=0.01)

cleanup()
logs = trainer.train_epoch(loader, alpha=1.0)
print(logs)