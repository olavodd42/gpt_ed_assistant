import os, json, pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from src.preprocess_functions import preprocess
from src.dataset_mm import EDDataset, collate_fn
from src.utils import to_py

# ==================== CONFIG BÁSICA ====================
TEXT_COL: str = "chiefcomplaint"
HASH_N: int = 20000
SVD_D: int  = 256
RANDOM_STATE: int = 42

# Faixas plausíveis (triagem e vitais)
VITAL_BOUNDS: Dict[str, Tuple[int, int]] = {
    "triage_temperature": (30, 45),
    "triage_heartrate": (20, 220),
    "triage_resprate": (4, 60),
    "triage_o2sat": (50, 100),
    "triage_sbp": (50, 260),
    "triage_dbp": (20, 150),
    "triage_pain": (0, 10),

    "ed_temperature_last": (30, 45),
    "ed_heartrate_last": (20, 220),
    "ed_resprate_last": (4, 60),
    "ed_o2sat_last": (50, 100),
    "ed_sbp_last": (50, 260),
    "ed_dbp_last": (20, 150),
    "ed_pain_last": (0, 10),
}

# Colunas candidatas a log1p (cauda pesada)
LOG1P_CANDS: List[str] = [
    "n_ed_30d","n_ed_90d","n_ed_365d",
    "n_hosp_30d","n_hosp_90d","n_hosp_365d",
    "n_icu_30d","n_icu_90d","n_icu_365d",
    "n_med","n_medrecon"
]

# Colunas para exclusão
DROP_MISC: List[str] = [
    "index","subject_id","hadm_id","stay_id","intime","outtime","dod","admittime","dischtime","deathtime",
    "edregtime","edouttime","intime_icu","next_ed_visit_time","ed_los","time_to_icu_transfer"
]

TARGET_COL: str = "outcome_critical"

try:
    df_train: pd.DataFrame = pd.read_csv("./data/interim/train.csv")
    df_valid: pd.DataFrame = pd.read_csv("./data/interim/valid.csv")
    df_test: pd.DataFrame  = pd.read_csv("./data/interim/test.csv")

    
    (Xnum_tr, Xcat_tr, Xtxt_tr, y_tr,
    Xnum_va, Xcat_va, Xtxt_va, y_va,
    Xnum_te, Xcat_te, Xtxt_te, y_te,
    txt_dim, artifacts) =preprocess(df_train, df_valid, df_test, TARGET_COL,
                                    vital_bounds=VITAL_BOUNDS, log1p_cands=LOG1P_CANDS,
                                    hash_n=HASH_N, svd_d=SVD_D, text_col=TEXT_COL, drop_misc=DROP_MISC)
    
    ds_tr: EDDataset = EDDataset(Xnum=Xnum_tr, Xcat=Xcat_tr, y=y_tr, Xtxt=Xtxt_tr)
    ds_va: EDDataset = EDDataset(Xnum=Xnum_va, Xcat=Xcat_va, y=y_va, Xtxt=Xtxt_va)
    ds_te: EDDataset = EDDataset(Xnum=Xnum_te, Xcat=Xcat_te, y=y_te, Xtxt=Xtxt_te)

    dl_tr: DataLoader = DataLoader(ds_tr, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dl_va: DataLoader = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dl_te: DataLoader = DataLoader(ds_te, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    os.makedirs("artifacts/datasets", exist_ok=True)

    # Salvar splits (np.savez_compressed aceita kwargs)
    np.savez_compressed("artifacts/datasets/train_proc.npz",
        Xnum=Xnum_tr,            # float32 (N, F_num)
        y=y_tr,                  # (N,)
        # categóricas: empacotar cada coluna com nome estável:
        **{f"Xcat_{i}": arr for i, arr in enumerate(Xcat_tr)},
        Xtxt=Xtxt_tr if Xtxt_tr is not None else np.array([], dtype=np.float32),
        has_txt = Xtxt_tr is not None
    )

    np.savez_compressed("artifacts/datasets/valid_proc.npz",
        Xnum=Xnum_va, y=y_va,
        **{f"Xcat_{i}": arr for i, arr in enumerate(Xcat_va)},
        Xtxt=Xtxt_va if Xtxt_va is not None else np.array([], dtype=np.float32),
        has_txt = Xtxt_va is not None
    )

    np.savez_compressed("artifacts/datasets/test_proc.npz",
        Xnum=Xnum_te, y=y_te,
        **{f"Xcat_{i}": arr for i, arr in enumerate(Xcat_te)},
        Xtxt=Xtxt_te if Xtxt_te is not None else np.array([], dtype=np.float32),
        has_txt = Xtxt_te is not None
    )

    with open("artifacts/num_scaler.pkl", "wb") as f:
        pickle.dump(artifacts["num_scaler"], f)

    with open("artifacts/text_pipe.pkl", "wb") as f:
        pickle.dump(artifacts["text_pipe"], f)   # dict {"hv":..., "tfidf":..., "svd":...}

    with open("artifacts/cat_maps.json", "w", encoding="utf-8") as f:
       json.dump(to_py(artifacts["cat_maps"]), f, ensure_ascii=False, indent=2)

    with open("artifacts/meta.json", "w", encoding="utf-8") as f:
        meta = {
                "num_cols": artifacts["num_cols"],
                "cat_cols": artifacts["cat_cols"],
                "text_col": artifacts["text_col"],
                "cat_cards": artifacts.get("cat_cards"),
                "txt_dim": int(Xtxt_tr.shape[1]) if Xtxt_tr is not None else 0,
                "clamp_limits": artifacts.get("clamp_limits", {}),
            }
        meta.update({
                "num_imputer_method": artifacts.get("num_imputer_method"),
                "num_imputer_values": to_py(artifacts.get("num_imputer_values", {}))
            })
        json.dump(to_py(meta), f, ensure_ascii=False, indent=2)

except Exception as e:
    print(f"Error opening files: {e}")