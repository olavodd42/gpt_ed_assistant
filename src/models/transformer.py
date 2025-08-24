from src.utils.seed import seed
seed()

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple
from src.utils.tokenization import load_tokenizer
from src.models.mlp import MLP

class Transformer(nn.Module):
    def __init__(self, cpkt: str, baseline: bool=False):
        """
        Inicializa o modelo transformer.
        Parâmetros:
            - cpkt: str
            - path: str
            - baseline: bool
            """
        super().__init__
        self.model = AutoModel.from_pretrained(cpkt)
        self.cfg = self.model.config
        self.baseline = baseline
        self.hidden_size = self.cfg.hidden_size

        if not baseline:
            self.selector_mlp = MLP(hidden_size=self.hidden_size, n=12)

        self.classifier_mlp = MLP(hidden_size=self.hidden_size, n=2)

    @torch.no_grad
    def _last_valid_index(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Retorna o índice do último elemento excluindo PADs.
        Parâmetros:
            - input_ids: torch.LongTensor [B, T]
        Retorna:
            - last: torch.longTensor    [B]
            """
        pad_id = self.cfg.pad_token_id

        # Máscara de pad
        is_pad = (input_ids == pad_id).int()

        # Verifica sequências sem nenhum PAD
        no_pad = (is_pad.sum(dim=1)==0)

        # índice do último token válido ao longo de T [B - 1]
        last = is_pad.argmax(dim=1) - 1

        # Correcao para linhas sem PAD (id do último elemento válido é o último da linha) [T - 1]
        last[no_pad] = input_ids.size(1) - 1

        return last
    
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Realiza a selecão do próximo grupo de exames e classifica o resultado clínico.
        Parâmetros:
            - input_ids: torch.LongTensor [B, T]
            - attention_mask: torch.LongTensor [B, T]
        Retorna:
            - selector_logits: torch.LongTensor     [N_actions, 12]
            - classifier_logits: torch.LongTensor   [B, 2]
        """

        # Obtém o último estado oculto
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state   # [B, T, H]
        B,_,_ = last_hidden_state.shape

        # Obtém o índice do último elemento válido
        last_idx = self._last_valid_index(input_ids)    # [B]

        # Se baseline=True então realiza a classificacão do resultado e seleciona o próximo grupo de exames
        # Caso contrário, apenas seleciona o próximo grupo
        if self.baseline:
            # Obtém vetor H do último token válido
            h = last_hidden_state[torch.arrange(B), last_idx]   # [B, H]
            
            # Classifica se ocorre resultado crítico ou não
            return self.classifier_mlp(h)   # [B, 2]
        
        # Remove eos final, pois ele não deve ser utilizado como acão
        eos_id = self.cfg.eos_token_id
        eos_mask = (input_ids == eos_id)    # [B, T]
        eos_mask[torch.arange(B), last_idx] = False

        # Encontra as logits para cada um dos 12 grupos
        selector_logits = self.selector_mlp(last_hidden_state[eos_mask])    # [N_actions, 12]
        # Encontra logits para classificacão clínica
        classifier_logits = self.classifier_mlp(last_hidden_state[torch.arange(B), last_idx])   # [B, 2]

        return selector_logits, classifier_logits   # ([N_actions, 12], [B, 2])
    
    def train(self):
        self.model.train()

    def parameters(self):
        return self.model.parameters()
    
    def __call__(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        return self.forward(input_ids, attention_mask)