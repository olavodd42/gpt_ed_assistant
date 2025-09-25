import os, json, joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from numpy.typing import NDArray
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


from src.utils import load_split_npz_concat, get_cat_indices, choose_f1_threshold, eval_split

ART_DIR = Path("artifacts")
DS_DIR  = ART_DIR / "datasets"          # <- volta pro diretório onde o preprocess salvou
MODEL_DIR = ART_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

meta: Dict[str, Any] = json.load(open(ART_DIR / "meta.json", "r", encoding="utf-8"))
print("num_cols:", len(meta.get("num_cols", [])))
print("cat_cols:", len(meta.get("cat_cols", [])))
print("text_dim:", meta.get("txt_dim"))

X_tr, y_tr, n_num, n_txt, n_cat = load_split_npz_concat(DS_DIR / "train_proc.npz")
X_va, y_va, _,     _,     _     = load_split_npz_concat(DS_DIR / "valid_proc.npz")
X_te, y_te, _,     _,     _     = load_split_npz_concat(DS_DIR / "test_proc.npz")

cat_idx: List[int] = get_cat_indices(n_num, n_txt, n_cat)
print(f"Shape train: {X_tr.shape} (num={n_num}, txt={n_txt}, cat={n_cat})")
print(f"Categóricas em índices: {cat_idx[:10]}{'...' if len(cat_idx)>10 else ''}")

# Desbalanceamento
pos: int = int((y_tr == 1).sum())
neg: int = int((y_tr == 0).sum())
spw: float = float(max((neg / max(1, pos)), 1.0))
print(f"Positivos={pos}, Negativos={neg}, scale_pos_weight={spw:.2f}")

lgbm: LGBMClassifier = LGBMClassifier(
    objective="binary",
    learning_rate=0.03,
    n_estimators=4000,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=50,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1,
    # Lidar com desbalanceamento
    scale_pos_weight=spw,
    # Métricas internas (não essenciais, só para log)
    metric=["auc", "aucpr"],
)

callbacks = [
    early_stopping(stopping_rounds=200, first_metric_only=True),
    log_evaluation(period=0),   # silencioso; troque para 50 se quiser logs periódicos
]

lgbm.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric=["auc", "aucpr"],
    categorical_feature=cat_idx if cat_idx else None,
    callbacks=callbacks
)

best_iter = getattr(lgbm, "best_iteration_", None)
print(f"Melhor iteração: {best_iter if best_iter is not None else 'n/d'}")

p_va: NDArray[np.float32] = lgbm.predict_proba(X_va)[:, 1]
thr, f1v = choose_f1_threshold(y_va, p_va)
print(f"Limiar ótimo no VALID por F1: {thr:.3f} (F1={f1v:.4f})")

_ = eval_split("VALID", y_va, p_va, thr)
p_te: NDArray[np.float32] = lgbm.predict_proba(X_te)[:, 1]
te_metrics: Dict[str, float] = eval_split("TEST", y_te, p_te, thr)

joblib.dump(
    {
        "model": lgbm,
        "threshold": thr,
        "meta": {
            "n_num": n_num, "n_txt": n_txt, "n_cat": n_cat, "cat_idx": cat_idx,
            "metrics_test": te_metrics
        }
    },
    MODEL_DIR / "lgbm_baseline.joblib"
)
print(f"✅ Modelo salvo em {MODEL_DIR / 'lgbm_baseline.joblib'}")
print(f"Melhor iteração: {lgbm.best_iteration_}")