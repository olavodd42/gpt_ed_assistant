import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

def to_py(obj):
    if isinstance(obj, np.generic):           # np.int64, np.float32, etc.
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    return obj

def load_split_npz(path_npz, n_cats: int) -> Tuple[NDArray[np.float32], List[NDArray[np.int64]], Optional[NDArray[np.float32]], NDArray[np.number]]:
    data: Dict[str, Any] = np.load(path_npz, allow_pickle=False)
    Xnum: NDArray[np.float32] = data["Xnum"]
    y: NDArray[np.number] = data["y"]
    has_txt: bool = bool(data["has_txt"])
    Xtxt: Optional[NDArray[np.float32]] = data["Xtxt"] if has_txt else None
    Xcat_list: List[NDArray[np.int64]] = [data[f"Xcat_{i}"] for i in range(n_cats)]
    return Xnum, Xcat_list, Xtxt, y

def make_bce_with_logits_pos_weight(y_train_numpy: NDArray[np.number], device: str | torch.device) -> torch.nn.BCEWithLogitsLoss:
    # y_train_numpy: array 0/1 do CONJUNTO DE TREINO
    pos = (y_train_numpy == 1).sum()
    neg = (y_train_numpy == 0).sum()
    # evita divisão por zero
    pos = max(1, int(pos))
    neg = max(1, int(neg))
    pw = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def get_cat_indices(n_num: int, n_txt: int, n_cat_cols: int):
    """Índices de colunas categóricas no array final [num | txt | cats]."""
    if n_cat_cols == 0:
        return []
    start = n_num + n_txt
    return list(range(start, start + n_cat_cols))

def choose_f1_threshold(y_true, y_prob):
    pr, rc, thr = precision_recall_curve(y_true, y_prob)  # returns precision, recall, thresholds
    # precision_recall_curve dá len(thr) = len(pr) - 1; alinhar F1 em thr
    pr = pr[:-1]; rc = rc[:-1]
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = np.argmax(f1)
    return float(thr[i]), float(f1[i])

def eval_split(name, y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    print(f"[{name}] AUROC={auroc:.4f} | AUPRC={auprc:.4f} | F1@{thr:.3f}={f1:.4f}")
    return {"auroc": auroc, "auprc": auprc, "f1": f1}