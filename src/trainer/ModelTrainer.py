from src.utils.seed import seed
seed()

import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModel, AutoTokenizer
from src.models.transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self, cpkt: str, tokenizer: AutoTokenizer, opt: torch.optim.adamw.AdamW, scaler: torch.cuda.amp.GradScaler,
                 clf_loss: torch.Tensor, sel_loss: torch.Tensor=None, baseline: bool=False):
        """
        Inicializa a classe.
        Parâmetros:
            - cpkt: str
            - tokenizer: transformers.AutoTokenizer
            - opt: torch.optim.adamw.Adamw
            - scaler: torch.cuda.amp.GradScaler
            - clf_loss: torch.Tensor
            - sel_loss: torch.Tensor
            - baseline: bool
            """
        self.tokenizer = tokenizer
        self.model = Transformer(cpkt, baseline=baseline).to(DEVICE)
        self.baseline = baseline
        self.cls_loss = clf_loss
        if not self.baseline:
            self.sel_loss = sel_loss

        self.opt = opt
        self.scaler = scaler

    def train(self, train_loader: torch.utils.data.DataLoader, alpha: float):
        """
        Treina o modelo.
        Parâmetros:
            - train_loader: torch.utils.data.DataLoader
        """
        self.model.train()
        for batch in train_loader:
            # Obtém as entradas do modelo (input_ids, attention_mask, y)
            input_ids = batch.get("input_ids").to(DEVICE)
            attention_mask = batch.get("attention_mask").to(DEVICE)
            y = batch['label']  # [B]

            sel_loss = 0.
            if not self.baseline:
                # Caso baseline não seja verdadeira, obtém também os grupos laboratoriais e os logits de classificacao e selecão
                # Obtém os valores alvos (action_groups)
                action_groups = batch.get("action_group_ids")
                selector_logits, classifier_logits = self.model(input_ids, attention_mask)
                if action_groups is not None:
                    target = torch.tensor(action_groups, device=DEVICE, dtype=torch.long)
                    # Cria one-hot encoding
                    target_sel_mh = torch.zeros((target.numel, 12), device=DEVICE)  # [N_actions, 12]
                    target_sel_mh.scatter_(1, target.view(-1,1), 1.)
                    # Calcula o loss de selecão
                    sel_loss = self.sel_loss(selector_logits, target_sel_mh)
            else:
                # Caso contrário, obtém apenas logits de classificacão
                classifier_logits = self.model(input_ids, attention_mask)
            
            # Obtém o loss de classificacão
            cls_loss = self.cls_loss(classifier_logits, y)
            
            # Calcula loss total: LOSS = CLASSIFICATION_LOSS + ALPHA*SELECTION_LOSS
            self.loss = cls_loss + alpha*sel_loss
            self.opt.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()