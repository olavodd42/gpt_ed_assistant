import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray

from torch.cuda.amp import GradScaler          # <- import correto
from torch.utils.data import DataLoader

from src.dataset_mm import EDDataset, collate_fn
from src.tiny_ednet_v2 import TinyEDNetV2
from src.run import train
from src.utils import load_split_npz_parts, make_bce_with_logits_pos_weight  # <- deve retornar (Xnum, Xcat_list, Xtxt, y)

# ==================== CONFIG ====================
DEVICE: str  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = 32 if DEVICE == "cuda" else 2048   # algo plausível
NUM_WORKERS: int = 4
PIN_MEMORY: bool = (DEVICE == "cuda")

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
    # ---------- META ----------
    meta_path = Path("artifacts/meta.json")
    meta: Dict[str, Any] = json.load(open(meta_path, "r", encoding="utf-8"))
    num_cols: List[str]  = meta.get("num_cols", [])
    cat_cols: List[str]  = meta.get("cat_cols", [])
    cat_cards: List[int] = meta.get("cat_cards", [])
    num_dim: int         = len(num_cols)
    txt_dim: int         = int(meta.get("txt_dim", 0))

    # ---------- DADOS ----------
    Xnum_tr, Xcat_tr, Xtxt_tr, y_tr = load_split_npz_parts("artifacts/datasets/train_proc.npz")
    Xnum_va, Xcat_va, Xtxt_va, y_va = load_split_npz_parts("artifacts/datasets/valid_proc.npz")
    Xnum_te, Xcat_te, Xtxt_te, y_te = load_split_npz_parts("artifacts/datasets/test_proc.npz")

    # Garantir dtypes consistentes
    Xnum_tr = Xnum_tr.astype(np.float32); y_tr = y_tr.astype(np.int64)
    Xnum_va = Xnum_va.astype(np.float32); y_va = y_va.astype(np.int64)
    Xnum_te = Xnum_te.astype(np.float32); y_te = y_te.astype(np.int64)
    if Xtxt_tr is not None: Xtxt_tr = Xtxt_tr.astype(np.float32)
    if Xtxt_va is not None: Xtxt_va = Xtxt_va.astype(np.float32)
    if Xtxt_te is not None: Xtxt_te = Xtxt_te.astype(np.float32)
    Xcat_tr = [c.astype(np.int64) for c in Xcat_tr]
    Xcat_va = [c.astype(np.int64) for c in Xcat_va]
    Xcat_te = [c.astype(np.int64) for c in Xcat_te]

    ds_tr = EDDataset(Xnum=Xnum_tr, Xcat=Xcat_tr, y=y_tr, Xtxt=Xtxt_tr)
    ds_va = EDDataset(Xnum=Xnum_va, Xcat=Xcat_va, y=y_va, Xtxt=Xtxt_va)
    ds_te = EDDataset(Xnum=Xnum_te, Xcat=Xcat_te, y=y_te, Xtxt=Xtxt_te)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)

    print("Data loaded successfully!")
    print(f"num_cols={num_dim} | cat_cols={len(cat_cols)} | txt_dim={txt_dim}")

    # ---------- MODELO ----------
    # Checagens úteis
    assert len(cat_cards) == len(cat_cols), f"cat_cards({len(cat_cards)}) != cat_cols({len(cat_cols)})"
    model = TinyEDNetV2(
        num_dim=num_dim,
        cat_cards=cat_cards,
        txt_dim=txt_dim,
        num_hidden=128,
        cat_hidden=128,
        txt_hidden=128,
        fusion_hidden=192,
        p_drop=0.2
    ).to(DEVICE)

    # ---------- OTIMIZAÇÃO ----------
    criterion = make_bce_with_logits_pos_weight(y_tr, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # AMP scaler: só habilita no CUDA
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # ---------- SMOKE TEST (opcional) ----------
    first_batch = next(iter(dl_tr))
    smoke_test_batch(model, first_batch, device=DEVICE)

    # ---------- TREINO ----------
    res = train(
        model,
        train_loader=dl_tr,
        val_loader=dl_va,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=50,
        use_amp=(DEVICE == "cuda"),
        scaler=scaler,
        optimizer_scheduler=scheduler,
        patience_epochs=10,
        grad_clip=1.0,
        save_path="artifacts/mlp_model.pt"
    )
    print(res)

except Exception as e:
    print(f"Error: {e}")
