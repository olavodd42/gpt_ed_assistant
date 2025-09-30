#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# ---------- paths ----------
DATA_PATH = "/home/olavo-dalberto/gpt_ed_assistant/data/processed"
MASTER_PATH = os.path.join(DATA_PATH, "..", "raw", "master.csv")
LAB_MAP_PATH = os.path.join(DATA_PATH, "..", "raw", "hosp", "lab_item_frequency_group.csv")  # ajuste se necessário
SPLITS = ["train.csv", "valid.csv", "test.csv"]

# outcomes como no split.py anterior
OUTCOME_COLS = ["outcome_critical", "outcome_ed_los", "lab_group_idx"]

# nomes legíveis para EHR/triagem
EHR_LABELS = {
    "age": "age",
    "gender": "sex",
    "chiefcomplaint": "chief complaint",
    "triage_temperature": "temp",
    "triage_heartrate": "hr",
    "triage_resprate": "rr",
    "triage_o2sat": "o2sat",
    "triage_sbp": "sbp",
    "triage_dbp": "dbp",
    "triage_pain": "pain",
    "triage_acuity": "acuity",
}

def load_lab_label_map(lab_map_path: str) -> dict:
    """
    Retorna dict: 'lab_<itemid>' -> 'nome legível do teste' (lowercase).
    Se não houver o CSV de mapeamento, usa fallback 'lab <itemid>'.
    """
    lab_name = {}
    try:
        m = pd.read_csv(lab_map_path, usecols=["itemid", "label"])
        m["itemid"] = m["itemid"].astype(str)
        m["label"] = m["label"].astype(str).str.lower().str.strip()
        for _, r in m.iterrows():
            lab_name[f"lab_{r['itemid']}"] = r["label"]
    except Exception as e:
        print(f"[WARN] não consegui ler {lab_map_path}: {e}\n"
              f"-> vou usar 'lab <itemid>' como nome.")
    return lab_name

def rebuild_feature_mapping(master_path: str, split_df: pd.DataFrame) -> list:
    """
    Reconstroi a lista de nomes originais (EHR + labs) na MESMA ordem usada no split:
    ed_ehr (na ordem definida) + sorted(lab_cols).
    Retorna lista ['age','gender',...,'lab_50802',...]
    """
    master = pd.read_csv(master_path)
    # EHR presentes em master (na ordem canônica abaixo)
    ed_ehr_order = [
        "age", "gender",
        "n_ed_30d", "n_ed_90d", "n_ed_365d",
        "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
        "n_icu_30d", "n_icu_90d", "n_icu_365d",
        "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
        "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
        "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
        "cci_Cancer2", "cci_HIV",
        "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
        "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
        "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
        "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
        "triage_temperature", "triage_heartrate", "triage_resprate",
        "triage_o2sat", "triage_sbp", "triage_dbp",
        "triage_pain", "triage_acuity", "chiefcomplaint",
    ]
    ed_ehr = [c for c in ed_ehr_order if c in master.columns]
    lab_cols = sorted([c for c in master.columns if c.startswith("lab_")])

    original_feats = ed_ehr + lab_cols
    # agora verifique se quantidade bate com número de features do split
    split_feats = [c for c in split_df.columns if c not in OUTCOME_COLS]
    if len(split_feats) != len(original_feats):
        raise RuntimeError(
            f"Quantidade de features do split ({len(split_feats)}) "
            f"≠ original ({len(original_feats)}). Recrie o split com o script atual."
        )
    return original_feats  # a ordem deve ser exatamente esta

def value_to_str(v):
    if pd.isna(v):
        return None
    # formatação compacta: 37.0 -> 37 ; 37.50 -> 37.5
    try:
        return f"{float(v):g}"
    except Exception:
        return str(v).strip()

def make_text_rows(df: pd.DataFrame, feat_cols: list, feat_names: list, lab_name_map: dict, only_labs: bool = False) -> pd.Series:
    # garante alinhamento: featurei -> feat_names[i]
    col2orig = {feat_cols[i]: feat_names[i] for i in range(len(feat_cols))}
    parts_all = []
    for _, row in df.iterrows():
        parts = []
        # EHR primeiro (se full)
        if not only_labs:
            for f in feat_cols:
                orig = col2orig[f]
                if orig.startswith("lab_"):
                    continue
                v = value_to_str(row[f])
                if v is None:
                    continue
                name = EHR_LABELS.get(orig, orig).lower()
                if orig == "chiefcomplaint":
                    v = str(v).lower()
                parts.append(f"{name} : {v}")
        # Depois labs
        for f in feat_cols:
            orig = col2orig[f]
            if not orig.startswith("lab_"):
                continue
            v = value_to_str(row[f])
            if v is None:
                continue
            name = lab_name_map.get(orig, f"lab {orig.split('_',1)[1]}").lower()
            parts.append(f"{name} : {v}")
        parts_all.append(" | ".join(parts))
    return pd.Series(parts_all, index=df.index)


def main():
    lab_name_map = load_lab_label_map(LAB_MAP_PATH)
    for fname in SPLITS:
        path = os.path.join(DATA_PATH, fname)
        df = pd.read_csv(path)

        # reconstrói nomes originais (EHR + labs) na ordem do split
        original_feat_names = rebuild_feature_mapping(MASTER_PATH, df)

        # congela lista de features do split (antes de criar colunas text_*)
        feat_cols = [c for c in df.columns if c not in OUTCOME_COLS and not c.startswith("text_")]
        if len(feat_cols) != len(original_feat_names):
            raise RuntimeError(f"n_feats split={len(feat_cols)} ≠ n_feats orig={len(original_feat_names)}")

        # gera colunas textuais
        df["text_labs"]  = make_text_rows(df, feat_cols, original_feat_names, lab_name_map, only_labs=True)
        df["text_full"]  = make_text_rows(df, feat_cols, original_feat_names, lab_name_map, only_labs=False)

        out_name = os.path.splitext(fname)[0] + "_linearized.csv"
        df.to_csv(os.path.join(DATA_PATH, out_name), index=False)
        print(f"[OK] {out_name} salvo ({len(df)} linhas).")

if __name__ == "__main__":
    main()
