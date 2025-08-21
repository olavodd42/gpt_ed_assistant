import os
import pandas as pd
import numpy as np
from ast import literal_eval
from src.utils import seed

seed()

# -----------------------------
# Config e leitura enxuta
# -----------------------------
PATH_MIMIC = "/home/olavo-dalberto/gpt_ed_assistant/data/raw"
PATH_PROC  = os.path.join(PATH_MIMIC, "processed")

df_master = pd.read_csv(
    os.path.join(PATH_PROC, 'master_dataset.csv'),
    dtype={"subject_id": "Int64", "hadm_id": "string", "stay_id": "string"},
    low_memory=False
)

# for col in ["subject_id", "hadm_id", "stay_id"]:
#     labs_wide[col] = pd.to_numeric(labs_wide[col], errors="coerce").round().astype("Int64")

for col in ["subject_id", "hadm_id", "stay_id"]:
    df_master[col] = pd.to_numeric(df_master[col], errors="coerce").round().astype("Int64")


# Apenas hospitalizados (como seu pipeline)
df_master_hosp = df_master[df_master["outcome_hospitalization"] == True].copy()

# master_lab_results com listas (já deduplicado anteriormente)
df_master_lab = pd.read_csv(
    os.path.join(PATH_PROC, 'master_lab_results.csv'),
    converters={'lab_measure': literal_eval, 'lab_results': literal_eval},
    dtype={"subject_id":"int32","hadm_id":"int64","stay_id":"int64"}
)

# Dicionário de itens -> label/grupo
df_items = pd.read_csv(
    os.path.join(PATH_MIMIC, "hosp", "lab_item_frequency_group.csv"),
    dtype={"itemid":"int32", "grouping":"string", "label":"string"}
).drop_duplicates("itemid")

# Normaliza para minúsculas e remove espaços extras
df_items["label"] = df_items["label"].str.lower().str.strip()
df_items["grouping"] = df_items["grouping"].str.lower().str.strip()

# Mapeamento de grupos para índice (padroniza chaves em lower)
group2idx = {
    'cbc':0, 'chem':1, 'coag':2, 'ua':3, 'lactate':4,
    'lfts':5, 'lipase':6, 'lytes':7, 'cardio':8,
    'blood gas':9, 'tox':10, 'inflammation':11
}

# -----------------------------
# Explode + join + pivot (sem iterrows)
# -----------------------------
# Explode as listas para linhas (muito mais rápido e estável)
dfe = df_master_lab.explode(["lab_measure", "lab_results"], ignore_index=True)

# Tipos compactos
dfe["lab_measure"] = dfe["lab_measure"].astype("int32")
dfe["lab_results"] = pd.to_numeric(dfe["lab_results"], errors="coerce").astype("float32")

# Junta metadados do item (label/grupo)
dfe = dfe.merge(df_items[["itemid","label","grouping"]], left_on="lab_measure", right_on="itemid", how="left")
dfe.drop(columns=["itemid"], inplace=True)

# Remove registros sem label (por segurança)
dfe = dfe.dropna(subset=["label"])

# Pivot para ter 1 coluna por label, preenchendo 0 (como seu código fazia)
# use 'first' porque você já filtrou duplicatas no CSV anterior
pivot = dfe.pivot_table(
    index=["subject_id","hadm_id","stay_id"],
    columns="label",
    values="lab_results",
    aggfunc="first"
).astype("float32").fillna(0.0)

# As colunas viram nível simples
pivot.columns.name = None
pivot = pivot.reset_index()

# -----------------------------
# lab_group_idx por stay (lista de índices únicos)
# -----------------------------
# mapeia grouping->idx (lower) com fallback para descartar o que não conhecer
dfe["group_idx"] = dfe["grouping"].map(lambda g: group2idx.get(str(g).lower(), None))

# agrega grupos distintos em lista ordenada
lab_groups = (dfe.dropna(subset=["group_idx"])
                .groupby(["subject_id","hadm_id","stay_id"], sort=False)["group_idx"]
                .agg(lambda s: sorted(set(map(int, s))))
                .reset_index(name="lab_group_idx"))

# junta pivot (valores de lab) + grupos
labs_wide = pivot.merge(lab_groups, on=["subject_id","hadm_id","stay_id"], how="left")

# remove pacientes sem nenhum grupo comum
labs_wide = labs_wide[labs_wide["lab_group_idx"].map(lambda x: isinstance(x, list) and len(x) > 0)]

# Garanta os mesmos dtypes nas chaves dos dois DataFrames

# -----------------------------
# Merge com master de hospitalizados
# -----------------------------
df_master_hosp_lab = df_master_hosp.merge(
    labs_wide, on=["subject_id","hadm_id","stay_id"], how="right"
)

# -----------------------------
# Filtros e limpeza de vitais
# -----------------------------
print('Before filtering for "age" >= 18 : master dataset size = ', len(df_master_hosp_lab))
df_master_hosp_lab = df_master_hosp_lab[df_master_hosp_lab['age'] >= 18]
print('After  filtering for "age" >= 18 : master dataset size = ', len(df_master_hosp_lab))

print('Before filtering for non-null "triage_acuity": ', len(df_master_hosp_lab))
df_master_hosp_lab = df_master_hosp_lab[df_master_hosp_lab['triage_acuity'].notnull()]
print('After  filtering for non-null "triage_acuity": ', len(df_master_hosp_lab))

# -----------------------------
# Temperatura -> Celsius (vetorizado)
# -----------------------------
# Se suas colunas de vitais seguem padrão "t0_temperature", "t1_temperature", etc:
temp_cols = [c for c in df_master_hosp_lab.columns if c.endswith("_temperature")]
if temp_cols:
    # Fahrenheit -> Celsius: (x - 32) * 5/9
    df_master_hosp_lab[temp_cols] = (df_master_hosp_lab[temp_cols] - 32.0) * (5.0/9.0)

# -----------------------------
# Outliers (vetorizado)
# -----------------------------
vitals_valid_range = {
    'temperature': {'outlier_low': 14.2, 'valid_low': 26,  'valid_high': 45,  'outlier_high': 47},
    'heartrate':   {'outlier_low': 0,    'valid_low': 0,   'valid_high': 350, 'outlier_high': 390},
    'resprate':    {'outlier_low': 0,    'valid_low': 0,   'valid_high': 300, 'outlier_high': 330},
    'o2sat':       {'outlier_low': 0,    'valid_low': 0,   'valid_high': 100, 'outlier_high': 150},
    'sbp':         {'outlier_low': 0,    'valid_low': 0,   'valid_high': 375, 'outlier_high': 375},
    'dbp':         {'outlier_low': 0,    'valid_low': 0,   'valid_high': 375, 'outlier_high': 375},
    'pain':        {'outlier_low': 0,    'valid_low': 0,   'valid_high': 10,  'outlier_high': 10},
    'acuity':      {'outlier_low': 1,    'valid_low': 1,   'valid_high': 5,   'outlier_high': 5},
}

# todas as colunas numéricas de vitais (exceto 'acuity')
vitals_cols = [c for c in df_master_hosp_lab.columns
               if "_" in c and c.split("_", 1)[1] in vitals_valid_range and c.split("_",1)[1] != "acuity"]

for col in vitals_cols:
    vtype = col.split("_", 1)[1]
    vr = vitals_valid_range[vtype]
    x = df_master_hosp_lab[col].astype("float32")

    # fora da faixa outlier -> NaN
    mask_out = (x < vr['outlier_low']) | (x > vr['outlier_high'])
    x = x.mask(mask_out, np.nan)

    # entre (outlier_low, valid_low) e (valid_high, outlier_high) -> clip para limite válido
    x = x.clip(lower=vr['valid_low'], upper=vr['valid_high'])

    df_master_hosp_lab[col] = x

# Imputação mediana vetorizada
if vitals_cols:
    imp = SimpleImputer(strategy='median')
    df_master_hosp_lab[vitals_cols] = imp.fit_transform(df_master_hosp_lab[vitals_cols])

# -----------------------------
# Label de LOS>24h e salvar
# -----------------------------
df_master_hosp_lab["outcome_ed_los"] = df_master_hosp_lab["ed_los_hours"] > 24

out_path = os.path.join(PATH_PROC, "master.csv")
df_master_hosp_lab.to_csv(out_path, index=False)
print("Salvo:", out_path, "| linhas =", len(df_master_hosp_lab))
