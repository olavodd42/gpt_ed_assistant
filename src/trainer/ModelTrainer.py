from src.utils.seed import seed
seed()

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Any, Optional, Iterable
from transformers import AutoTokenizer
from src.models.transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self,
                 cpkt: str,
                 tokenizer: AutoTokenizer,
                 #opt: torch.optim.Optimizer,
                 clf_loss: torch.nn.Module,
                 sel_loss: Optional[torch.nn.Module] =None,
                 baseline: bool=False,
                 use_amp: bool = True,
                 grad_clip: float = 1.0,
                 selector_num_classes: int = 12,
                 scaler: Optional[GradScaler] = None,
                 accumulation_steps: int = 1
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
        self.baseline = baseline
        self.model = Transformer(cpkt, tokenizer=self.tokenizer, baseline=self.baseline).to(DEVICE)
        self.clf_loss_fn = clf_loss
        self.sel_loss_fn = sel_loss if not self.baseline else None
        self.use_amp = use_amp and DEVICE.type == "cuda"
        self.scaler = scaler if (self.use_amp and scaler is not None) else (GradScaler() if self.use_amp else None)
        self.grad_clip = grad_clip
        self.selector_num_classes = selector_num_classes
        self.opt = None
        self.accumulation_steps = accumulation_steps

        if self.sel_loss_fn is None:
            self.sel_loss_fn = self.clf_loss_fn

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def configure_optimizer(self, opt_fn, **opt_kwargs):
        """Cria o otimizador com os parâmetros do modelo."""
        self.opt = opt_fn(self._trainable_params(), **opt_kwargs)

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
        assert self.opt is not None, "Chame configure_optimizer(...) antes!"
        self.model.train()

        total_loss = 0.0
        total_batches = 0
        step_in_accum = 0

        for batch_idx, batch in enumerate(train_loader):
            # Obtém as entradas do modelo (input_ids, attention_mask, y)
            input_ids = batch.get("input_ids").to(DEVICE, non_blocking=True)
            attention_mask = batch.get("attention_mask").to(DEVICE, non_blocking=True)
            y = batch['label']  # [B]
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            y = y.to(DEVICE, non_blocking=True).long() # CE espera torch.longTensor

            # Evita acúmulo de gradientes apenas quando for rodados todos os batches acumulados
            if step_in_accum == 0:
                self.opt.zero_grad(set_to_none=True)

            # forward pass
            if self.use_amp:
                # Realiza operacões com FP16
                with autocast(dtype=torch.float16):
                    outputs = self.model(input_ids, attention_mask)

                    # Se baseline = True, então realiza apenas classificacão
                    if self.baseline:
                        clf_logits = outputs
                        clf_loss = self.clf_loss_fn(clf_logits, y)
                        sel_loss = input_ids.new_zeros((), dtype=torch.float32) 
                    else:
                        sel_logits, clf_logits = outputs
                        clf_loss = self.clf_loss_fn(clf_logits, y)
                        action_groups = batch["action_group_ids"]

                        if sel_logits.numel() > 0 and ("action_group_ids" in batch) and (action_groups is not None):
                            # Obtém os valores da variável alvo (grupo)
                            sel_target = self._build_selector_targets(action_groups, n_actions=sel_logits.shape[0])
                            sel_loss = self.sel_loss_fn(sel_logits, sel_target)
                        else:
                            sel_loss = input_ids.new_zeros((), dtype=torch.float32)
                
                    loss = clf_loss + alpha*sel_loss
                    # Normalização pelo número de acúmulos
                    loss = loss / self.accumulation_steps
                
                # backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_in_accum += 1
                last_step = (batch_idx == len(train_loader) - 1)
                if (step_in_accum == self.accumulation_steps) or last_step:
                    if self.use_amp:
                        self.scaler.unscale_(self.opt)
                        
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.use_amp:
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        self.opt.step()
                        

                    step_in_accum = 0

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

                loss /= self.accumulation_steps

                # backward sem scaler
                loss.backward()
                step_in_accum += 1

                if (step_in_accum == self.accumulation_steps) or (batch_idx == len(train_loader) - 1):
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.step()
                    step_in_accum = 0

            total_loss += float(loss.detach().cpu())
            total_batches += 1

        return {"loss": total_loss / max(total_batches, 1)}