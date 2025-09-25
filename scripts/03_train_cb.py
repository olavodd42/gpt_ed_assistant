import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from src.utils import load_split_npz_parts, choose_f1_threshold, eval_split

# Carrega PARTS
Xnum_tr, Xcat_tr, Xtxt_tr, y_tr = load_split_npz_parts("artifacts/datasets/train_proc.npz")
Xnum_va, Xcat_va, Xtxt_va, y_va = load_split_npz_parts("artifacts/datasets/valid_proc.npz")
Xnum_te, Xcat_te, Xtxt_te, y_te = load_split_npz_parts("artifacts/datasets/test_proc.npz")

# Garantir dtypes
y_tr = y_tr.astype(np.int32)
y_va = y_va.astype(np.int32)
y_te = y_te.astype(np.int32)

def assemble_df(Xnum, Xtxt, Xcat_list):
    parts = []
    # numéricos (float32)
    df_num = pd.DataFrame(Xnum.astype(np.float32),
                          columns=[f"num_{i}" for i in range(Xnum.shape[1])])
    parts.append(df_num)

    # texto (float32) — continua como numérico
    if Xtxt is not None and Xtxt.size > 0:
        df_txt = pd.DataFrame(Xtxt.astype(np.float32),
                              columns=[f"txt_{i}" for i in range(Xtxt.shape[1])])
        parts.append(df_txt)

    # categóricas (int32) — manter inteiras!
    if Xcat_list:
        Xcats = np.column_stack([c.astype(np.int32) for c in Xcat_list])
        df_cat = pd.DataFrame(Xcats, columns=[f"cat_{i}" for i in range(len(Xcat_list))])
        parts.append(df_cat)

    df = pd.concat(parts, axis=1)
    return df

X_tr_df = assemble_df(Xnum_tr, Xtxt_tr, Xcat_tr)
X_va_df = assemble_df(Xnum_va, Xtxt_va, Xcat_va)
X_te_df = assemble_df(Xnum_te, Xtxt_te, Xcat_te)

# Índices das colunas categóricas no DataFrame final
n_num = Xnum_tr.shape[1]
n_txt = 0 if (Xtxt_tr is None or Xtxt_tr.size == 0) else Xtxt_tr.shape[1]
n_cat = len(Xcat_tr)
cat_idx = list(range(n_num + n_txt, n_num + n_txt + n_cat))

# Desbalanceamento
pos = int((y_tr == 1).sum())
neg = int((y_tr == 0).sum())
spw = max(neg / max(1, pos), 1.0)

# Pools (pode passar o DataFrame diretamente)
train_pool = Pool(X_tr_df, y_tr, cat_features=cat_idx)
valid_pool = Pool(X_va_df, y_va, cat_features=cat_idx)
test_pool  = Pool(X_te_df, y_te, cat_features=cat_idx)

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    iterations=5000,
    subsample=0.9,
    rsm=0.9,
    random_seed=42,
    # Se sua versão não suportar scale_pos_weight, use:
    # class_weights=[1.0, spw]
    scale_pos_weight=spw,
    od_type="Iter",      # early stopping
    od_wait=200,
    verbose=False
)

model.fit(train_pool, eval_set=valid_pool)

# Threshold pelo F1 no VALID
p_va = model.predict_proba(valid_pool)[:, 1]
thr, _ = choose_f1_threshold(y_va, p_va)
eval_split("VALID", y_va, p_va, thr)

p_te = model.predict_proba(test_pool)[:, 1]
eval_split("TEST", y_te, p_te, thr)
