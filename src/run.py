import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, List, Tuple
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_auprc(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device | str,
    use_amp: bool = True
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Avalia o modelo no val_loader e retorna (AUPRC, probs, y_true).
    Parâmetros:
        * model: nn.Module -> o modelo Pytorch.
        * val_loader: DataLoader -> dataloader de validação.
        * device: torch.device | str -> se validação ocorre em "cuda" ou "gpu".
        * use_amp: bool -> se True, então é utilizado **mixed precision** para acelerar a GPU (padrão é True).
    Retorno:
        * ap: float -> métrica AUPRC (**Averaged Precision**), ou seja, a área sob a curva Precision-Recall.
        * probs: torch.Tensor -> tensor dos scores de probabilidade.
        * y_true: torch.Tensor -> tensor dos rótulos verdadeiros.
    """

    model.eval()
    all_probs: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            x_num: torch.Tensor = batch["x_num"].to(device, non_blocking=True)
            x_cat: List[torch.Tensor] = [t.to(device, non_blocking=True) for t in batch["x_cat_list"]]
            x_txt: Optional[torch.Tensor] = batch.get("x_txt", None)
            if x_txt is not None:
                x_txt = x_txt.to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()

            with autocast(device_type=device, enabled=use_amp):
                logits: torch.Tensor = model(x_num, x_cat, x_txt)
                probs: torch.Tensor = torch.sigmoid(logits)

            all_probs.append(probs.detach().cpu())
            all_y.append(y.detach().cpu())

    probs_cat: torch.Tensor = torch.cat(all_probs)  # (N_val,)
    y_cat: torch.Tensor = torch.cat(all_y)          # (N_val,)

    # AUPRC padrão (binária): usa y_true binário e scores contínuos
    try:
        ap: float = float(average_precision_score(y_cat.numpy().astype(int), probs_cat.numpy()))
    except Exception:
        ap = float("nan")  # em casos patológicos (p.ex., classe única)

    try:
        auc: float = float(roc_auc_score(y_cat.numpy().astype(int), probs_cat.numpy()))
    except Exception:
        auc = float("nan")  # em casos patológicos (p.ex., classe única)
    return ap, auc, probs_cat, y_cat


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device | str,
    num_epochs: int = 50,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
    optimizer_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    patience_epochs: int = 10,
    grad_clip: float = 1.0,
    save_path: str = "artifacts/best_model.pt",
) -> Dict[str, Any]:
    """
    Treina o modelo com early stopping por AUPRC de validação.
    Parâmetros:
        * model: nn.Module -> o modelo Pytorch.
        * train_loader: DataLoader -> o dataloader de treino.
        * val_loader: DataLoader -> o dataloader de validação.
        * criterion: nn.Module -> função de loss.
        * optimizer: optim.Optimizer -> otimizador.
        * device: torch.device | str -> se treino ocorre em "cuda" ou "gpu".
        * num_epochs: int -> número de épocas do treino (padrão é 50).
        * use_amp: bool -> se True, então é utilizado **mixed precision** para acelerar a GPU (padrão é True).
        * scaler: Optional[GradScaler] -> GradScaler para amp (padrão é None).
        * optimizer_scheduler: Optional[optim.lr_scheduler._LRScheduler] -> scheduler opcional (padrão é None).
        * patience_epochs: int -> paciência do early stopping (padrão é 10).
        * save_path: str -> caminho para salvar o melhor state_dict.

    Retorna Dict com:
        * best_ap: float -> melhor valor de **Averaged Precision (AP)** encontrado.
        * best_epoch: int -> época que obteve o melhor **AP**.
        * history: List[Dict[str, float]] -> dict com o train_loss e valid_aucpr de cada época.
    """
    model.to(device)
    if scaler is None:
        scaler = GradScaler(device=device, enabled=use_amp)

    best_ap: float = -1.0
    best_epoch: int = -1
    waited: int = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum: float = 0.0
        train_count: int = 0

        for batch in train_loader:
            x_num: torch.Tensor = batch["x_num"].to(device, non_blocking=True)
            x_cat: List[torch.Tensor] = [t.to(device, non_blocking=True) for t in batch["x_cat_list"]]
            x_txt: Optional[torch.Tensor] = batch.get("x_txt")
            if x_txt is not None:
                x_txt = x_txt.to(device, non_blocking=True)
            y: torch.Tensor = batch["y"].to(device, non_blocking=True).float()  # (B,)

            # Zera gradientes
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, enabled=use_amp):
                logits: torch.Tensor = model(x_num, x_cat, x_txt)  # (B,)
                loss: float = criterion(logits, y)         

            if use_amp:
                # Faz backward escalonado
                scaler.scale(loss).backward()

                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)  # Traz gradientes ao espaço real
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # Traz estabilidade numérica
                
                # Aplica o passo do otimizador e atualiza a escala para próxima iteração
                scaler.step(optimizer)
                scaler.update()
            else:
                # Backward sem escalonador
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            batch_size: int = y.size(0)
            # Acumula loss corrigido pelo batch_size
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        # Obtém o loss médio da época
        train_loss: float = train_loss_sum / max(1, train_count)

        # ---- validação (AUPRC) ----
        ap, auc, probs_val, y_val = evaluate_auprc(model, val_loader, device, use_amp=use_amp)

        # ---- scheduler ----
        if optimizer_scheduler is not None:
            # compatível com ReduceLROnPlateau e schedulers comuns
            if hasattr(optimizer_scheduler, "step") and \
               optimizer_scheduler.__class__.__name__.lower().startswith("reduce"):
                optimizer_scheduler.step(ap)
            else:
                optimizer_scheduler.step()

        print(f"ep {epoch:02d} | train_loss={train_loss:.4f} | valid_auc={auc:.4f} | valid_AUPRC={ap:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "valid_auc": auc, "valid_auprc": ap})

        # ---- early stopping ----
        if ap > best_ap:
            best_ap = ap
            best_auc = auc
            best_epoch = epoch
            waited = 0
            torch.save(model.state_dict(), save_path)
        else:
            waited += 1
            if patience_epochs is not None and waited >= patience_epochs:
                print("Early stopping!")
                break

    return {"best_ap": best_ap, "best_auc": best_auc, "best_epoch": best_epoch, "history": history}
