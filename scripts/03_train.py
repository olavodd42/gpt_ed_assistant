import json
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from src.dataset_mm import EDDataset, collate_fn
from src.tiny_ednet_v2 import TinyEDNetV2
from src.utils import load_split_npz, make_bce_with_logits_pos_weight

DEVICE: str  = "cuda" if torch.cuda.is_available() else "cpu"

def smoke_test_batch(model, batch, device="cuda"):
    model = model.to(device).eval()
    with torch.no_grad():
        x_num = batch["x_num"].to(device)
        x_cat = [t.to(device) for t in batch["x_cat_list"]]
        x_txt = batch.get("x_txt")
        if x_txt is not None:
            x_txt = x_txt.to(device)
        logits = model(x_num, x_cat, x_txt)
        assert logits.ndim == 1, f"logits ndim!=1, got {logits.shape}"
        print("Smoke test OK:", logits.shape)

        

try:
    meta: Dict[str, Any]    = json.load(open("artifacts/meta.json", "r", encoding="utf-8"))
    n_cats: int             = len(meta.get("cat_cols", []))
    num_cols: List[str]     = meta.get("num_cols", [])
    cat_cols: List[str]     = meta.get("cat_cols", [])
    cat_cards: List[int]    = meta.get("cat_cards", [])
    num_dim: int            = len(num_cols)
    txt_dim: int            = int(meta.get("txt_dim", 0))

    train_data: Tuple[NDArray[np.float32], List[NDArray[np.int64]], Optional[NDArray[np.float32]], NDArray[np.number]]  = load_split_npz("artifacts/datasets/train_proc.npz", n_cats)
    valid_data: Tuple[NDArray[np.float32], List[NDArray[np.int64]], Optional[NDArray[np.float32]], NDArray[np.number]]  = load_split_npz("artifacts/datasets/valid_proc.npz", n_cats)
    test_data: Tuple[NDArray[np.float32], List[NDArray[np.int64]], Optional[NDArray[np.float32]], NDArray[np.number]]   = load_split_npz("artifacts/datasets/test_proc.npz",  n_cats)
    
    Xnum_tr: NDArray[np.float32]            = train_data[0]
    Xcat_tr: List[NDArray[np.int64]]        = train_data[1]
    Xtxt_tr: Optional[NDArray[np.float32]]  = train_data[2]
    y_tr: NDArray[np.number]                = train_data[3]
    Xnum_va: NDArray[np.float32]            = valid_data[0]
    Xcat_va: List[NDArray[np.int64]]        = valid_data[1]
    Xtxt_va: Optional[NDArray[np.float32]]  = valid_data[2]
    y_va: NDArray[np.number]                = valid_data[3]
    Xnum_te: NDArray[np.float32]            = test_data[0]
    Xcat_te: List[NDArray[np.int64]]        = test_data[1]
    Xtxt_te: Optional[NDArray[np.float32]]  = test_data[2]
    y_te: NDArray[np.number]                = test_data[3]

    ds_tr: EDDataset    = EDDataset(Xnum=Xnum_tr, Xcat=Xcat_tr, y=y_tr, Xtxt=Xtxt_tr)
    ds_va: EDDataset    = EDDataset(Xnum=Xnum_va, Xcat=Xcat_va, y=y_va, Xtxt=Xtxt_va)
    ds_te: EDDataset    = EDDataset(Xnum=Xnum_te, Xcat=Xcat_te, y=y_te, Xtxt=Xtxt_te)

    print("Data loaded succesfully!")

    
    print(f"num_cols={len(num_cols)} | cat_cols={len(cat_cols)} | txt_dim={txt_dim}")


    model: TinyEDNetV2  = TinyEDNetV2(
        num_dim=num_dim,
        cat_cards=cat_cards,
        txt_dim=txt_dim,
        num_hidden=128,
        cat_hidden=128,
        txt_hidden=128,
        fusion_hidden=192,
        p_drop=0.2
    )

    model   = model.to(DEVICE)
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, collate_fn=collate_fn,
                   num_workers=4, pin_memory=True)
    
    batch = next(iter(dl_tr))
    
    x_num = batch["x_num"].to(DEVICE, non_blocking=True)                 # (B, F_num)
    x_cat = [t.to(DEVICE, non_blocking=True) for t in batch["x_cat_list"]]  # [ (B,), ... ]
    x_txt = batch.get("x_txt")
    if x_txt is not None:
        x_txt = x_txt.to(DEVICE, non_blocking=True)
    y = batch["y"].to(DEVICE, non_blocking=True).float()   

    # logits = model(x_num, x_cat, x_txt)  # (B,)
    # criterion = make_bce_with_logits_pos_weight(y_tr)
    # loss = criterion(logits, y)
    # print(f"LOSS = {loss.item():.3E}")

    model.eval()  # sem dropout/bn variando
    with torch.no_grad():
        logits = model(x_num, x_cat, x_txt)    # (B,)
        print("logits:", logits.shape, logits.dtype, logits.device)

    criterion = make_bce_with_logits_pos_weight(y_tr).to(DEVICE)  # garanta device!
    loss = criterion(logits, y)
    print(f"LOSS = {loss.item():.6f}")

    with torch.no_grad():
        probs = torch.sigmoid(logits)          # (B,)
        preds = (probs >= 0.5).long()          # threshold simples
        print("probs range:", float(probs.min()), "→", float(probs.max()))
        print("preds sum #positivos:", int(preds.sum()))
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

    y_np = y.detach().cpu().numpy().astype(int)
    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    print("ACC:",  accuracy_score(y_np, preds_np))
    print("F1 :",  f1_score(y_np, preds_np, zero_division=0))
    # Para AUROC/AUPRC use as probabilidades (contínuas), não os rótulos binários:
    try:
        print("AUROC:", roc_auc_score(y_np, probs_np))
    except ValueError:
        print("AUROC: falhou (talvez todas as classes iguais neste batch)")

    try:
        print("AUPRC:", average_precision_score(y_np, probs_np))
    except ValueError:
        print("AUPRC: falhou (talvez todas as classes iguais neste batch)")

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = make_bce_with_logits_pos_weight(y_tr).to(DEVICE)  # garanta device!

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_num, x_cat, x_txt)
    loss = criterion(logits, y)
    print(f"loss (antes) = {loss.item():.6f}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # opcional, estável
    optimizer.step()

    # mede após 1 passo (com model.eval para evitar dropout interferir)
    model.eval()
    with torch.no_grad():
        logits2 = model(x_num, x_cat, x_txt)
        loss2 = criterion(logits2, y)
    print(f"loss (depois) = {loss2.item():.6f}")

    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False, collate_fn=collate_fn,
                   num_workers=4, pin_memory=True)

    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

    model.eval()
    all_probs, all_y, all_preds = [], [], []
    with torch.no_grad():
        for batch in dl_va:
            x_num = batch["x_num"].to(DEVICE, non_blocking=True)
            x_cat = [t.to(DEVICE, non_blocking=True) for t in batch["x_cat_list"]]
            x_txt = batch.get("x_txt")
            if x_txt is not None:
                x_txt = x_txt.to(DEVICE, non_blocking=True)
            yb = batch["y"].to(DEVICE, non_blocking=True).float()

            logits = model(x_num, x_cat, x_txt)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_y.append(yb.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_y     = torch.cat(all_y).numpy().astype(int)

    print("Valid ACC :", accuracy_score(all_y, all_preds))
    print("Valid F1  :", f1_score(all_y, all_preds, zero_division=0))
    print("Valid AUROC:", roc_auc_score(all_y, all_probs))
    print("Valid AUPRC:", average_precision_score(all_y, all_probs))
    smoke_test_batch(model, batch, DEVICE)

except Exception as e:
    print(f"Error: {e}")