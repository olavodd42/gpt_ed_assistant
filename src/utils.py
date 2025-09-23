import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Any, Optional

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

def make_bce_with_logits_pos_weight(y_train_numpy):
    # y_train_numpy: array 0/1 do CONJUNTO DE TREINO
    pos = (y_train_numpy == 1).sum()
    neg = (y_train_numpy == 0).sum()
    # evita divisão por zero
    pos = max(1, int(pos))
    neg = max(1, int(neg))
    pw = torch.tensor(neg / pos, dtype=torch.float32)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw)
