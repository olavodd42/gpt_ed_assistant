import numpy as np
import torch
from numpy.typing import NDArray
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

def to_py(obj: Any) -> Any:
    if isinstance(obj, np.generic):           # np.int64, np.float32, etc.
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    return obj

# def load_split_npz(path_npz: str, n_cats: int) -> Tuple[NDArray[np.float32], List[NDArray[np.int64]], Optional[NDArray[np.float32]], NDArray[np.number]]:
#     data: Dict[str, Any] = np.load(path_npz, allow_pickle=False)
#     Xnum: NDArray[np.float32] = data["Xnum"]
#     y: NDArray[np.number] = data["y"]
#     has_txt: bool = bool(data["has_txt"])
#     Xtxt: Optional[NDArray[np.float32]] = data["Xtxt"] if has_txt else None
#     Xcat_list: List[NDArray[np.int64]] = [data[f"Xcat_{i}"] for i in range(n_cats)]
#     return Xnum, Xcat_list, Xtxt, y

def _read_npz(path_npz: str | Path) -> Dict[str, Any]:
    return np.load(str(path_npz), allow_pickle=False)

def load_split_npz_parts(path_npz: str | Path,
                         n_cats: Optional[int] = None
                         ) -> Tuple[
                            NDArray[np.float32],                # Xnum
                            List[NDArray[np.int64]],            # Xcat_list
                            Optional[NDArray[np.float32]],      # Xtxt
                            NDArray[np.number]                  # y
                         ]:
    """
    Para MLP / EDDataset.
    Retorna (Xnum, Xcat_list, Xtxt, y). Autodetecta Xcat_i, mas você pode forçar via n_cats.
    """
    data = _read_npz(path_npz)
    Xnum = data["Xnum"].astype(np.float32)
    y    = data["y"]                  # dtype pode ser int/float; quem usa decide
    Xtxt = None
    if bool(data["has_txt"]) and "Xtxt" in data.files and data["Xtxt"].size > 0:
        Xtxt = data["Xtxt"].astype(np.float32)

    Xcat_list: List[NDArray[np.int64]] = []
    if n_cats is None:
        i = 0
        while f"Xcat_{i}" in data.files:
            Xcat_list.append(data[f"Xcat_{i}"].astype(np.int64))
            i += 1
    else:
        for i in range(n_cats):
            Xcat_list.append(data[f"Xcat_{i}"].astype(np.int64))

    return Xnum, Xcat_list, Xtxt, y

def load_split_npz_concat(path_npz: str | Path
                          ) -> Tuple[
                              NDArray[np.float32],   # X concatenado [num | txt | cats]
                              NDArray[np.int32],     # y (int32)
                              int, int, int          # n_num, n_txt, n_cat_cols
                          ]:
    """
    Para LightGBM / modelos baseados em árvores.
    Retorna (X, y, n_num, n_txt, n_cat_cols) com X = [num | txt | cats].
    """
    data = _read_npz(path_npz)

    # Numéricos
    Xnum = data["Xnum"].astype(np.float32)
    n_num = Xnum.shape[1]

    # Alvo
    y = data["y"].astype(np.int32)

    # Texto (opcional)
    Xtxt = None
    n_txt = 0
    if bool(data["has_txt"]) and "Xtxt" in data.files and data["Xtxt"].size > 0:
        Xtxt = data["Xtxt"].astype(np.float32)
        n_txt = int(Xtxt.shape[1])

    # Categóricas
    Xcat_list: List[NDArray[np.int32]] = []
    i = 0
    while f"Xcat_{i}" in data.files:
        Xcat_list.append(data[f"Xcat_{i}"].astype(np.int32))
        i += 1
    n_cat_cols = len(Xcat_list)

    blocks: List[NDArray[np.float32]] = [Xnum]
    if Xtxt is not None:
        blocks.append(Xtxt)
    if n_cat_cols > 0:
        Xcats = np.column_stack(Xcat_list).astype(np.float32)
        blocks.append(Xcats)

    X = np.column_stack(blocks).astype(np.float32)
    return X, y, n_num, n_txt, n_cat_cols

def make_bce_with_logits_pos_weight(y_train_numpy: NDArray[np.number], device: str | torch.device) -> torch.nn.BCEWithLogitsLoss:
    # y_train_numpy: array 0/1 do CONJUNTO DE TREINO
    pos = (y_train_numpy == 1).sum()
    neg = (y_train_numpy == 0).sum()
    # evita divisão por zero
    pos = max(1, int(pos))
    neg = max(1, int(neg))
    pw = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def get_cat_indices(n_num: int, n_txt: int, n_cat_cols: int) -> List[int]:
    """Índices de colunas categóricas no array final [num | txt | cats]."""
    if n_cat_cols == 0:
        return []
    start = n_num + n_txt
    return list(range(start, start + n_cat_cols))


def choose_f1_threshold(y_true: NDArray[np.number], y_prob: NDArray[np.float64]) -> Tuple[float, float]:
    pr, rc, thr = precision_recall_curve(y_true, y_prob)  # returns precision, recall, thresholds
    # precision_recall_curve dá len(thr) = len(pr) - 1; alinhar F1 em thr
    pr = pr[:-1]; rc = rc[:-1]
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def eval_split(name: str, y_true: NDArray[np.number], y_prob: NDArray[np.float64], thr: float) -> Dict[str, float]:
    y_pred: NDArray[np.int64] = (y_prob >= thr).astype(int)
    auroc: float = roc_auc_score(y_true, y_prob)
    auprc: float = average_precision_score(y_true, y_prob)
    f1: float = f1_score(y_true, y_pred)
    print(f"[{name}] AUROC={auroc:.4f} | AUPRC={auprc:.4f} | F1@{thr:.3f}={f1:.4f}")
    return {"auroc": auroc, "auprc": auprc, "f1": f1}