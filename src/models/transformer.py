
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, hidden_size: int, n: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n),
        )
    def forward(self, x):
        return self.net(x)
    

class Transformer(nn.Module):
    def __init__(self, cpkt: str,tokenizer: AutoTokenizer, baseline: bool=False):
        """
        Modelo base (encoder) + duas cabeças:
        - selector_mlp: prevê 12 grupos de exames (multi-label, BCE)
        - classifier_mlp: prevê 2 classes (CE)
        Se baseline=True, só usa a cabeça de classificação.
        """

        super().__init__()

        self.model = AutoModel.from_pretrained(cpkt, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # Congela encoder
        self.model.requires_grad_(False)
        # Desativa dropout do encoder
        self.model.eval()
        self.cfg = self.model.config
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("Tokenizer sem eos_token_id. Garanta add_special_tokens({'eos_token':'<eos>'}).")
        if pad_id is None:
            pad_id = eos_id
        self.cfg.eos_token_id = eos_id
        self.cfg.pad_token_id = pad_id
        self.model.resize_token_embeddings(len(tokenizer))
        self.baseline = baseline
        self.hidden_size = self.cfg.hidden_size

        if not baseline:
            self.selector_mlp = MLP(hidden_size=self.hidden_size, n=12)

        self.classifier_mlp = MLP(hidden_size=self.hidden_size, n=2)

        if getattr(self.cfg, "pad_token_id", None) is None:
            raise ValueError("pad_token_id ausente no config do modelo base.")
        if getattr(self.cfg, "eos_token_id", None) is None and not baseline:
            raise ValueError("eos_token_id ausente no config; necessário para o selector.")

    @torch.no_grad
    def _last_valid_index(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Índice do último token NÃO-PAD por linha.
        Parâmetros:
            - input_ids: torch.LongTensor [B, T] com PAD = pad_token_id
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

        # quando não há PAD na linha, queira o último índice (T-1)
        T = input_ids.size(1)
        last[no_pad] = T - 1

        return last
    
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Realiza a selecão do próximo grupo de exames e classifica o resultado clínico.
        Parâmetros:
            - input_ids: torch.LongTensor [B, T]
            - attention_mask: torch.LongTensor [B, T]
        Se baseline=True, retorna:
            - classifier_logits: [B,2]
        Caso contrário, retorna:
            - selector_logits: [N_actions, 12] (um logit por grupo de exame em cada <eos> interno)
            - classifier_logits: [B,2] (no último token válido de cada sequência)
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
            last_h = last_hidden_state[torch.arange(B, device=input_ids.device), last_idx]   # [B, H]
            
            # Classifica se ocorre resultado crítico ou não
            return self.classifier_mlp(last_h)   # [B, 2]
        
        # Seleção de ações em <eos> internos
        eos_id = self.cfg.eos_token_id
        eos_mask = (input_ids == eos_id)    # [B, T]
        eos_mask[torch.arange(B, device=input_ids.device), last_idx] = False    # remove o <eos> final

        first_eos_idx = eos_mask.int().argmax(dim=1)  # [B]
        eos_mask[torch.arange(B, device=input_ids.device), first_eos_idx] = False

        print("eos per sample:", eos_mask.sum(dim=1).tolist())

        # Encontra as logits para cada um dos 12 grupos
        selector_logits = self.selector_mlp(last_hidden_state[eos_mask])    # [N_actions, 12]
        # Encontra logits para classificacão clínica
        classifier_logits = self.classifier_mlp(last_hidden_state[torch.arange(B, device=input_ids.device), last_idx])   # [B, 2]

        return selector_logits, classifier_logits   # ([N_actions, 12], [B, 2])
    
    # def train(self):
    #     self.model.train()

    # def parameters(self):
    #     return self.model.parameters()
    
    # def __call__(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
    #     return self.forward(input_ids, attention_mask)