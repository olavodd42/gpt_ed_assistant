import torch
import torch.nn as nn
from torch.cuda.amp import autocast

def train_one_epoch(model: nn.Module,
                    loader,
                    criterion,
                    optimizer,
                    scaler,
                    device="cuda",
                    grad_clip=1.0,
                    use_amp=True
                    ):
    """
    Realiza o treino de uma época.
    Parâmetros:
        * model: nn.Module -> o modelo FTTransformer.
    """
    model.train()
    total_loss = 0.0
    n_obs = 0
    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = [t.to(device) for t in batch["x_cat_list"]]
        x_txt = batch["x_txt"]
        x_txt = x_txt.to(device) if (x_txt is not None) else None
        y = batch["y"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x_num, x_cat, x_txt)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n_obs += bs
    return total_loss / max(1, n_obs)
