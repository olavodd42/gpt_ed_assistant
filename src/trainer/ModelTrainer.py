from src.utils.seed import seed
seed()

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Iterable
from transformers import AutoTokenizer
from src.models.transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self,
                 cpkt: str,
                 tokenizer: AutoTokenizer,
                 opt: torch.optim.Optimizer,
                 clf_loss: torch.nn.Module,
                 sel_loss: Optional[torch.nn.Module] =None,
                 baseline: bool=False,
                 use_amp: bool = True,
                 grad_clip: float = 1.0,
                 selector_num_classes: int = 12,
                 scaler: Optional[GradScaler] = None
            ):
        """
        Inicializa a classe.
        Parâmetros:
            - cpkt: str
            - tokenizer: transformers.AutoTokenizer
            - opt: torch.optim.Optimizer
            - clf_loss: torch.nn.Module
            - sel_loss: Optional[torch.nn.Module]
            - baseline: bool
            - use_amp: bool
            - grad_clip: float
            - selector_num_classes: int
            - scaler: torch.cuda.amp.GradScaler
            """
        self.tokenizer = tokenizer
        self.model = Transformer(cpkt, baseline=baseline).to(DEVICE)
        self.baseline = baseline
        self.clf_loss_fn = clf_loss
        self.sel_loss_fn = sel_loss if not self.baseline else None
        self.opt = opt
        self.use_amp = use_amp and DEVICE.type == "cuda"
        self.scaler = scaler if (self.use_amp and scaler is not None) else (GradScaler() if self.use_amp else None)
        self.grad_clip = grad_clip
        self.selector_num_classes = selector_num_classes

        if self.sel_loss_fn is None and self.baseline:
            raise ValueError("Error: The selector loss function must not be none if self.baseline = True!")

    def _build_selector_targets(self, action_groups: Iterable[int], n_actions: int):
        """
        Parãmetros:
            - action_groups: Iterable[int]
            - n_actions: int - tamanho de action_groups_flat
        Retorna:
         - torch.Tensor [n_actions, 12] - one-hot: um grupo por ação
        """
        tgt = torch.as_tensor(list(action_groups), device=DEVICE, dtype=torch.long) # [n_actions]

        # Cria one-hot encoding
        target = torch.zeros((n_actions, self.selector_num_classes), device=DEVICE, dtype=torch.float32)  # [N_actions, 12]
        target.scatter_(1, tgt.view(-1,1), 1.)
        return target

    def train_epoch(self, train_loader: Iterable, alpha: float = 1.):
        """
        Treina 1 época. 
        Parâmetros:
            - train_loader: Iterable. O DataLoader deve fornecer dict com:
                - input_ids: [B,T]
                - attention_mask: [B,T]
                - label: [B] (0/1)   -> CE
                - action_group_ids (opcional se baseline=False): lista achatada de ids de grupo (len = N_actions no batch)
            - alpha: float
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            # Obtém as entradas do modelo (input_ids, attention_mask, y)
            input_ids = batch.get("input_ids").to(DEVICE)
            attention_mask = batch.get("attention_mask").to(DEVICE)
            y = batch['label']  # [B]
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            y = y.to(DEVICE).long() # CE espera torch.longTensor

            # Evita acúmulo de gradientes
            self.opt.zero_grad(set_to_none=True)

            # forward pass
            if self.use_amp:
                # Realiza operacões com FP16/BF16 quando seguro
                with autocast():
                    outputs = self.model(input_ids, attention_mask)

                    # Se baseline = True, então realiza apenas classificacão
                    if self.baseline:
                        clf_logits = outputs
                        clf_loss = self.clf_loss_fn(clf_logits)
                        sel_loss = torch.tensor(0., device=DEVICE)
                    else:
                        sel_logits, clf_logits = outputs
                        clf_loss = self.clf_loss_fn(clf_logits, y)
                        action_groups = batch["action_group_ids"]

                        if sel_logits.numel() > 0 and ("action_group_ids" in batch) and (action_groups is not None):
                            # Obtém os valores da variável alvo (grupo)
                            sel_target = self._build_selector_targets(action_groups, n_actions=sel_logits.shape[0])
                            sel_loss = self.sel_loss_fn(sel_logits, sel_target)
                        else:
                            sel_loss = torch.tensor(0., device=DEVICE)
                
                    loss = clf_loss + alpha*sel_loss
                
                if self.scaler is not None:
                    # backward com scaler
                    # scaler evita underflow em fp16 e clip_grad_norm_ evita estouro do gradiente
                    self.scaler.scale(loss).backward
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()

            # Se self.use_amp = False -> sem autocast e sem scaler
            else:
                # forward sem autocast
                outputs = self.model(input_ids, attention_mask)

                if self.baseline:
                    clf_logits = outputs
                    clf_loss = self.clf_loss_fn(clf_logits, y)
                    sel_loss = torch.tensor(0., device=DEVICE)

                else:
                    sel_logits, clf_logits = outputs
                    clf_loss = self.clf_loss_fn(clf_logits, y)
                    action_groups = batch["action_group_ids"]

                    if sel_logits.numel() > 0 and ("action_group_ids" in batch) and (action_groups is not None):
                        # Obtém os valores da variável alvo (grupo)
                        sel_target = self._build_selector_targets(action_groups, n_actions=sel_logits.shape[0])
                        sel_loss = self.sel_loss_fn(sel_logits, sel_target) 

                    else:
                        sel_loss = torch.tensor(0.0, device=DEVICE)

                loss = clf_loss + alpha*sel_loss

                # backward sem scaler
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

            total_loss += loss.item()
            total_batches += 1

        return {"loss": total_loss / max(total_batches, 1)}