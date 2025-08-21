
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.utils.seed import seed

seed()
PATH_DATA = "/home/olavo-dalberto/gpt_ed_assistant/data/"
PATH_PROC  = os.path.join(PATH_DATA, "raw", "processed")
OUT_PATH = os.path.join(PATH_DATA, "processed")
df_master = pd.read_csv(os.path.join(PATH_PROC, "master.csv"))

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

# Seleciona colunas de interesse
df_master = df_master[ed_ehr + ed_lab + outcome_cols].copy()

# Garanta que os dois outcomes binários sejam 0/1 (coerção segura)
for c in ['outcome_critical','outcome_ed_los']:
    df_master[c] = pd.to_numeric(df_master[c], errors='coerce').fillna(0).astype(int)
    df_master[c] = df_master[c].clip(0,1)  # caso venha algo diferente

# NÃO toque em lab_group_idx (lista) — útil para análises depois

# (opcional) arredonda temperatura
if 'triage_temperature' in df_master.columns:
    df_master['triage_temperature'] = df_master['triage_temperature'].round(1)

# Mapa de features (interpretabilidade)
feature_names = [f'feature{i+1}' for i in range(len(ed_ehr + ed_lab))]
feature_map = dict(zip(feature_names, ed_ehr + ed_lab))
pd.Series(feature_map).to_json(os.path.join(OUT_PATH, "feature_map.json"), indent=2)

# Renomeia colunas (somente features), preserva outcomes originais
df_features = df_master[ed_ehr + ed_lab].copy()
df_features.columns = feature_names
df_outcomes = df_master[['outcome_critical','outcome_ed_los','lab_group_idx']]
df_master_renamed = pd.concat([df_features, df_outcomes], axis=1)

# -------- Estratificação: use APENAS os binários --------
strata = (
    df_master_renamed[['outcome_critical','outcome_ed_los']]
    .astype(int).astype(str).agg('_'.join, axis=1)
)

train_df, validtest_df = train_test_split(
    df_master_renamed, test_size=0.20, stratify=strata
)

strata_vt = (
    validtest_df[['outcome_critical','outcome_ed_los']]
    .astype(int).astype(str).agg('_'.join, axis=1)
)

valid_df, test_df = train_test_split(
    validtest_df, test_size=0.50, stratify=strata_vt
)

os.makedirs(OUT_PATH, exist_ok=True)
train_df.to_csv(os.path.join(OUT_PATH, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUT_PATH, "valid.csv"), index=False)
test_df.to_csv(os.path.join(OUT_PATH, "test.csv"), index=False)

print("Splits salvos:",
      len(train_df), len(valid_df), len(test_df),
      "| classes train:",
      train_df[['outcome_critical','outcome_ed_los']].value_counts().to_dict())
