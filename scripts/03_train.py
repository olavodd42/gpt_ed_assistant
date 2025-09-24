import json
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from src.dataset_mm import EDDataset, collate_fn
from src.tiny_ednet_v2 import TinyEDNetV2
from src.run import train
from src.utils import load_split_npz, make_bce_with_logits_pos_weight

DEVICE: str  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = 65536*2

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
    
    
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, collate_fn=collate_fn)

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
    criterion = make_bce_with_logits_pos_weight(y_tr, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    res = train(
        model,
        train_loader=dl_tr,
        val_loader=dl_va,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=50,
        use_amp=True,
        scaler=GradScaler(device=DEVICE, enabled=True),
        optimizer_scheduler=scheduler,
        patience_epochs=10,
        grad_clip=1.0,
        save_path="artifacts/best_model.pt"
    )
    print(res)

except Exception as e:
    print(f"Error: {e}")