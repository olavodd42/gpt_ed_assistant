import os
import pandas as pd
import numpy as np
from collections import defaultdict
from src.utils import seed

seed()
# ---------- Config ----------
PATH_MIMIC = "/home/olavo-dalberto/gpt_ed_assistant/data/raw"  # ajuste se necessário
MASTER_CSV = os.path.join(PATH_MIMIC, "processed", "master_dataset.csv")
LABEVENTS_CSV = os.path.join(PATH_MIMIC, "hosp", "labevents.csv")
DLAB_CSV = os.path.join(PATH_MIMIC, "hosp", "d_labitems_labeled.csv")
OUT_CSV = "/home/olavo-dalberto/gpt_ed_assistant/data/raw/processed/master_lab_results.csv"

CHUNKSIZE = 1_000_000  # ajuste conforme RAM; 1M é um bom começo
USECOLS_LAB = ["subject_id","hadm_id","itemid","charttime","storetime","valuenum"]
DTYPES_LAB = {
    "subject_id": "int32",
    "hadm_id": "float64",  # pode vir NaN na origem
    "itemid": "int32",
    "valuenum": "float32",
}
PARSE_DATES = ["storetime"]  # charttime é opcional aqui

# ---------- 1) Master (somente hospitalizados) ----------
df_master = pd.read_csv(MASTER_CSV)
df_master_hosp = df_master[df_master["outcome_hospitalization"] == True].copy()

# Mantenha só o necessário, e garanta datetime eficiente
df_master_hosp = df_master_hosp[["subject_id","hadm_id","stay_id","intime","outtime"]].copy()
df_master_hosp["intime"]  = pd.to_datetime(df_master_hosp["intime"])
df_master_hosp["outtime"] = pd.to_datetime(df_master_hosp["outtime"])

# Vamos usar esse set para filtrar rápido no labevents
hadm_keep = set(df_master_hosp["hadm_id"].astype("int64").tolist())

# ---------- 2) D_LABITEMS: lista de labs comuns ----------
df_labitems = pd.read_csv(DLAB_CSV, usecols=["itemid","ed_labs"])
labitems_common_list = df_labitems.loc[df_labitems["ed_labs"] == 1, "itemid"].astype("int32").tolist()
item_keep = set(labitems_common_list)

# ---------- 3) Agregadores incrementais ----------
# Para cada chave (subj, hadm, stay), acumulamos listas
agg_measures = defaultdict(list)  # itemid por stay
agg_results  = defaultdict(list)  # valuenum por stay

# Mapeamento rápido de (subject_id, hadm_id) -> todas as stays e janelas (pode ter várias stays por HADM)
# Usaremos merge para expandir no chunk; manter DF leve ajuda
master_keys = df_master_hosp[["subject_id","hadm_id","stay_id","intime","outtime"]].copy()
master_keys["hadm_id"] = master_keys["hadm_id"].astype("int64")  # para bater com cast do chunk

# ---------- 4) Stream de labevents ----------
reader = pd.read_csv(
    LABEVENTS_CSV,
    usecols=USECOLS_LAB,
    dtype=DTYPES_LAB,
    parse_dates=PARSE_DATES,
    chunksize=CHUNKSIZE,
    low_memory=True,
)

total_rows = 0
kept_rows = 0
merged_rows = 0

for i, chunk in enumerate(reader, 1):
    total_rows += len(chunk)

    # hadm_id pode vir float; converta e filtre rapidamente
    chunk = chunk[pd.notna(chunk["hadm_id"])]
    if chunk.empty:
        continue
    chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

    # Pré-filtros por hadm e item
    chunk = chunk[chunk["hadm_id"].isin(hadm_keep)]
    chunk = chunk[chunk["itemid"].isin(item_keep)]
    chunk = chunk[pd.notna(chunk["valuenum"])]
    if chunk.empty:
        continue
    kept_rows += len(chunk)

    # Merge com janelas do ED (pode expandir se houver várias stays por hadm)
    # Mantemos só colunas necessárias para filtrar por tempo:
    chunk = chunk.merge(
        master_keys,
        on=["subject_id","hadm_id"],
        how="inner",
        copy=False,
        validate="many_to_many",
    )
    if chunk.empty:
        continue

    # Filtrar "apenas labs no ED": intime <= storetime <= outtime
    # storetime pode estar ausente; garantimos parse na leitura
    m = (chunk["storetime"] <= chunk["outtime"]) & (chunk["intime"] <= chunk["storetime"])
    chunk = chunk[m]
    if chunk.empty:
        continue
    merged_rows += len(chunk)

    # Agregar incrementalmente (sem groupby.apply)
    # Ordenar é opcional; diminui variação entre execuções
    grp = chunk[["subject_id","hadm_id","stay_id","itemid","valuenum"]].sort_values(
        ["subject_id","hadm_id","stay_id","itemid"], kind="stable"
    )
    for (subj, hadm, stay), gdf in grp.groupby(["subject_id","hadm_id","stay_id"], sort=False):
        # Acrescenta listas deste chunk ao agregado global
        agg_measures[(subj, hadm, stay)].extend(gdf["itemid"].tolist())
        agg_results[(subj, hadm, stay)].extend(gdf["valuenum"].tolist())

    if i % 10 == 0:
        print(f"[chunk {i}] total={total_rows:,} kept_prefilter={kept_rows:,} after_window_merge={merged_rows:,}")

print(f"TOTAL lido={total_rows:,} | após prefiltros={kept_rows:,} | após janela ED={merged_rows:,}")

# ---------- 5) Montar DataFrame final ----------
keys = list(agg_measures.keys())
if len(keys) == 0:
    print("Nenhum registro elegível encontrado. Verifique filtros/paths.")
else:
    df_out = pd.DataFrame(keys, columns=["subject_id","hadm_id","stay_id"])
    df_out["lab_measure"] = [agg_measures[k] for k in keys]
    df_out["lab_results"] = [agg_results[k] for k in keys]

    # Marcar stays com duplicatas de itemid
    def has_duplicates(lst):
        # True se houver itemid repetido no mesmo stay
        return len(lst) != len(set(lst))

    df_out["has_duplicates"] = df_out["lab_measure"].apply(has_duplicates)

    # Filtrar stays sem duplicatas (equivale ao seu passo original)
    filtered = df_out[~df_out["has_duplicates"]].drop(columns=["has_duplicates"]).copy()

    # Salvar
    filtered.to_csv(OUT_CSV, index=False)
    print(f"Salvo: {OUT_CSV} | linhas={len(filtered):,}")
