import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def suggest_emb_dim(card: int) -> int:
    """
    Heurística prática que define automaticamente o tamanho das embeddings categóricas,
    balanceando capacidade de representação e uso de memória, sem precisar escolher
    manualmente para cada coluna.
    Parâmetros:
        * card: int -> *cardinalidade* de uma variável categórica.
    Retorna:
        * int -> tamanho da embedding."""
    return int(min(32, 2 * math.ceil(math.sqrt(max(1, card)))))

class MLPBlock(nn.Module):
    r"""
    Define um bloco denso reutilizável que aplica possui a arquitetura a seguir:
        Input -> LayerNorm -> Linear -> SiLU -> DropOut -> Output
    serve como a camada básica do modelo multimodal.
    Parâmetros:
        * dim_in: int -> tamanho da entrada do bloco (input).
        * dim_out: int -> número de neurônios da camada de saída (output).
        * p_drop: float -> valor do dropout (padrão = 0.2).

    Atributos:
        * self.norm: LayerNorm -> camada de normalização.
        * self.fc: Linear -> camada linear (*fully connected*).
        * self.act: SiLU -> camada de ativação SiLU (*Sigmoid Linear Unit*).
           \[
                \operatorname{SiLU}(x) = x \cdot \sigma(x)
                = \frac{x}{1 + e^{-x}}
            \]
        * self.do: Dropout -> camada de regularização
    Métodos:
        * self.forward(self, x: torch.Tensor) -> torch.Tensor.
    """
    def __init__(self, dim_in: int, dim_out: int, p_drop: float = 0.2) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(dim_in)
        self.fc: nn.Linear      = nn.Linear(dim_in, dim_out)
        self.act: nn.SiLU       = nn.SiLU()
        self.do: nn.Dropout     = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza um step do treinamento.
        Parâmetros:
            * x: torch.Tensor -> tensor de input do bloco. (B, dim_in).
        Retorna:
            * x: torch.Tensor -> tensor de output do bloco. (B, dim_out).
        """
        x = self.norm(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.do(x)
        return x    # (B, dim_out)
    
class TinyEDNetV2(nn.Module):
    """
    Modelo multimodal leve para:
      * x_num: Tensor (B, num_dim)
      * x_cat_list: List[Tensor] (cada um shape (B,))
      * x_txt: Tensor (B, txt_dim) ou None
    Parâmetros:
        * num_dim: int        -> número de variáveis (colunas) numéricas do dataset.
        * cat_card: List[int] -> cardinalidade por coluna categórica (inclui UNK).
                    cardinalidade = max_id + 1
        * txt_dim: int        -> tamanho do braço (camada) textual (0 se não houver).
        * num_hidden: int     -> largura do braço numérico (128 por padrão).
        * cat_hidden: int     -> largura após projetar embeddings (128 por padrão).
        * txt_hidden: int     -> largura do braço textual (128 por padrão).
        * fusion_hidden: int  -> largura do braço de fusão, inclui residual (192 por padrão).
        * p_drop: float       -> dropout global dos blocos (0.2 por padrão).

    Atributos:
        * self.has_txt: bool                -> True se há coluna de texto, senão False.
        * self.n_cats: int                  -> número de features categóricas.
        * self.num_proj: MLPBlock           -> bloco numérico.
        * self.cat_embs: nn.ModuleList      -> lista de embeddings gerados (uma por coluna).
        * self.cat_proj: Optional[MLPBlock] -> bloco categórico.
        * self.txt_norm: LayerNorm          -> camada de normalização do bloco textual (se houver).
        * self.txt_fc: Linear               -> camada linear do bloco textual (se houver).
        * self.txt_do: Dropout              -> camada de regularização do bloco textual (se houver).
        * self.fuse_norm: LayerNorm         -> camada de normalização do bloco de fusão.
        * self.fuse_c1: Linear              -> camada linear 1 do bloco de fusão.
        * self.fuse_do1: Dropout            -> camada de regularização 1 do bloco de fusão.
        * self.fuse_fc2: Linear             -> camada linear 2 do bloco de fusão.
        * self.head: Sequential             -> modelo sequencial de cabeça do modelo. Estrutura:
            LayerNorm(fused_in) → Linear(fused_in → H) → SiLU → Dropout → Linear(H → 1)
    """
    def __init__(
        self,
        num_dim: int,
        cat_cards: List[int],     
        txt_dim: int = 0,
        num_hidden: int = 128,    
        cat_hidden: int = 128,         
        txt_hidden: int = 128,      
        fusion_hidden: int = 192,    
        p_drop: float = 0.2
    ) -> None:
        super().__init__()

        self.has_txt: bool = (txt_dim is not None) and (txt_dim > 0)
        self.n_cats: int  = len(cat_cards)

        # ----- braço numérico -----
        self.num_proj: MLPBlock = MLPBlock(num_dim, num_hidden, p_drop=p_drop)

        # ----- braço categórico -----
        self.cat_embs: nn.ModuleList = nn.ModuleList([nn.Embedding(card, suggest_emb_dim(card)) for card in cat_cards])
        emb_total = 0
        for card in cat_cards:
            emb_dim = suggest_emb_dim(card)  # sua heurística
            self.cat_embs.append(nn.Embedding(card, emb_dim))
            emb_total += emb_dim
        print(f"[DEBUG] n_cats={self.n_cats}, emb_total={emb_total}, cat_hidden={cat_hidden}")

        self.cat_proj: Optional[MLPBlock] = MLPBlock(emb_total, cat_hidden, p_drop=p_drop) if self.n_cats > 0 else None

        # ----- braço texto -----
        if self.has_txt:
            self.txt_norm: nn.LayerNorm = nn.LayerNorm(txt_dim)
            self.txt_fc: nn.Linear      = nn.Linear(txt_dim, txt_hidden)
            self.txt_do: nn.Dropout     = nn.Dropout(p_drop)

        # ----- fusão + cabeça -----
        fused_in: int = num_hidden + (cat_hidden if self.n_cats > 0 else 0) + (txt_hidden if self.has_txt else 0)

        self.fuse_norm: nn.LayerNorm = nn.LayerNorm(fused_in)
        self.fuse_fc1: nn.Linear  = nn.Linear(fused_in, fusion_hidden)
        self.fuse_do1: nn.Dropout  = nn.Dropout(p_drop)
        self.fuse_fc2: nn.Linear  = nn.Linear(fusion_hidden, fused_in)  # residual volta à mesma dimensão
        H = max(64, fusion_hidden // 2)
        # Recebe a representação fundida (h) e gera um único logit por paciente para classificação
        self.head: nn.Sequential = nn.Sequential(
            # (fused_in,) -> (fused_in,)
            nn.LayerNorm(fused_in),
            # (fused_in,) -> (H, )
            nn.Linear(fused_in, H),
            nn.SiLU(),
            nn.Dropout(p_drop),
            # (H,) -> (1,)
            nn.Linear(H, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Incializa os pesos da seguinte forma:
            * **Linear**: distribuição normal truncada (sigma=0.02) e bias zero.
            * **Embedding**: distribuição normal (sigma=0.02).
        Garante que todos braços fiquem estáveis e gradientes fluem bem.
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat_list: List[torch.Tensor], x_txt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Esta função:
            1. Extrai representações latentes separadas (numérico, categórico, texto).
            2. Concatena e passa por um bloco de fusão residual.
            3. Usa a head para reduzir para 1 logit por paciente.

        Esse logit é a predição do modelo antes de aplicar a sigmoid (probabilidade de outcome crítico).
        Parâmetros:
            * x_num: torch.Tensor -> features numéricas contínuas normalizadas.
            * x_cat_list: List[torch.Tensor] -> lista de features categoricas mapeadas em IDs.
            * x_txt: Optional[torch.Tensor] -> vetor de texto reduzido, se houver (padrão: None).
        Retorna:
            * logit: torch.Tensor -> vetor contendo os logits de cada registro.
        
        """
        # NUMÉRICAS
        # INPUT = x_num: (B, F_num)
        # MLPBlock(num_dim -> num_hidden)
        h_num: torch.Tensor = self.num_proj(x_num)  
        # OUTPUT = h_num: (B, num_hidden)

        # CATEGÓRICAS
        if self.n_cats > 0:
            embs: List[torch.Tensor] = []
            for emb, ids in zip(self.cat_embs, x_cat_list):
                # INPUT = ids: (B,), dtype long, com 0=UNK
                embs.append(emb(ids))       # (B, emb_dim)
            h_cat: Optional[torch.Tensor] = torch.cat(embs, dim=1)  # (B, sum_embs)
            # MLPBlock(sum_embs → cat_hidden)
            if self.cat_proj is not None:
                h_cat = self.cat_proj(h_cat)
            # OUTPUT = h_cat: (B, cat_hidden)
        else:
            h_cat = None

        # texto
        h_txt: Optional[torch.Tensor] = None
        if self.has_txt and x_txt is not None:
            # INPUT = x_txt: (B, txt_dim)
            t: torch.Tensor = self.txt_norm(x_txt)
            t = self.txt_fc(t)
            t = F.silu(t)
            t = self.txt_do(t)
            h_txt = t
            # OUTPUT = h_txt: (B, txt_hidden)

        # fusão
        parts: List[torch.Tensor] = [h_num]
        if h_cat is not None:
            parts.append(h_cat)
        if h_txt is not None:
            parts.append(h_txt)
        
        # MLP: LayerNorm -> Linear -> Dropout -> Linear
        h: torch.Tensor = torch.cat(parts, dim=1)         # (B, fused_in)
        z: torch.Tensor = self.fuse_norm(h)
        z = F.silu(self.fuse_fc1(z))
        z = self.fuse_do1(z)
        z = self.fuse_fc2(z)
        # Conexão residual
        h = h + z                            # residual

        # fused_in -> 1
        logit: torch.Tensor = self.head(h).squeeze(-1)     # (B,)
        return logit