import torch
import torch.nn as nn
from typing import List, Optional

class FTTransformer(nn.Module):
    """
    Implementação minimalista inspirada em Gorishniy et al.:
        - Numéricos: cada feature -> token via projeção (peso d_token x 1) + bias por feature.
        - Texto SVD: tratado como numérico (concatena aos numéricos).
        - Categóricas: Embedding(cardinalidade, d_token) por coluna.
        - Sequência: [CLS] + tokens_num + tokens_cat
        - Encoder: TransformerEncoder (prenorm)
        - Cabeçalho: LN -> Linear -> 1 logit

    Parâmetros:
        * n_num: int -> número de features numéricas.
        * cat_cards: List[int] -> cardinalidades das features categóricas.
        * d_token: int -> dimensão do token, múltiplo de n_heads (padrão é 192).
        * n_layers: int -> número de camadas do encoder (padrão é 4).
        * n_heads: int -> número de cabeças de atenção (padrão é 8).
        * ff_mult: int -> multiplicador da largura da camada feed-forward interna, que é ff_mult * d_token (padrão é 4).
        * p_drop_token: float -> dropout do nível da sequência de tokens (padrão é 0.0)
        * p_drop_attn: float -> dropout do encoder (padrão é 0.1)
    
    Atributos da classe:
        * self.n_num: int -> número de features numéricas.
        * self.n_cat: int -> número de features categóricas.
        * self.d: int -> dimensão do token, múltiplo de n_heads.
        * self.num_weight: nn.Parameter -> matriz de pesos W (n_num, d).
        * self.num_bias: nn.Parameter -> matriz bias do modelo B (n_num, d).
        * self.cat_embs: nn.ModuleList -> lista de camadas de embeddings das colunas categóricas.
        * self.cls_token: nn.Parameter -> vetor [CLS] (1, 1, d).
        * self.drop_tokens: nn.Dropout -> camada de dropout para aplicar na sequência de tokens.
        * self.encoder: nn.TransformerEncoder -> empilha n_layer camadas encoders idênticas.
        * self.head: nn.Sequential -> cabeça da classificação, normaliza o vetor do [CLS] e projeta para 1 logit.
    """

    def __init__(
        self,
        n_num: int,
        cat_cards: List[int],
        d_token: int = 192,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        p_drop_token: float = 0.0,
        p_drop_attn: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_num: int = n_num
        self.n_cat: int = len(cat_cards)
        self.d: int = d_token

        # Tokenizador de numéricos
        # token_j = W_j * x_j + B_j
        self.num_weight: nn.Parameter = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias: nn.Parameter   = nn.Parameter(torch.zeros(n_num, d_token))
        nn.init.xavier_uniform_(self.num_weight)

        # Embeddings categóricas (um por coluna)
        self.cat_embs: nn.ModuleList = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cards
        ])

        # Token [CLS]
        self.cls_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Dropout "de token" (opcional)
        self.drop_tokens: nn.Dropout = nn.Dropout(p_drop_token)

        # Encoder Transformer (prenorm)
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_token,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,       # (B, L, d)
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cabeçalho
        self.head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )

    def _num_to_tokens(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Faz unsqueeze na variável numérica x_num e usa broadcast para multiplicar pela
        matriz de weights e somar ao bias.
        Parâmetros:
            * x_num: torch.Tensor -> tensor de float 32.    (B, n_num)
        Retorna:
            * torch.Tensor -> vetor gerado ao multiplicar por num_weight e somar com num_bias.  (B, n_num, d)
        """
        return x_num.unsqueeze(-1) * self.num_weight + self.num_bias
    
    def _cat_to_tokens(self, x_cat_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Aplica as camadas de embeddings a cada coluna categórica e empilha.
        Parâmetros:
            * x_cat_list: List[torch.Tensor] -> lista de tensores Long das colunas categóricas geradas.     [(B,), ...]

        Retorna:
            * Optional[torch.Tensor] -> tensores de embeddings gerados empilhados, ou None.  (B, n_cat, d)
        """
        if not x_cat_list:
            return None
        cat_tokens: List[torch.Tensor] = []
        for t, emb in zip(x_cat_list, self.cat_embs):
            # t: (B,) int64/long
            cat_tokens.append(emb(t)) 

        return torch.stack(cat_tokens, dim=1)
    
    def forward(self,
                x_num: torch.Tensor,
                x_cat_list: List[torch.Tensor],
                x_txt: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Faz um passo de treino.
        Parâmetros:
            * x_num: torch.Tensor -> tensor das variáveis numéricas.    (B, n_num)
            * x_cat_list: List[torch.Tensor] -> lista de tensores numéricos das variáveis categóricas.    [(B,),...]
            * x_txt: Optional[torch.Tensor] -> tensor dos embeddings textuais ou None.
        Funcionamento:
            1) Concatena tensores de texto e numérico.  (B, n_num_total)
            2) Converte numéricos em tokens.    (B, n_num_total, d)
            3) Converte categóricas em tokens, se houver.   (B, n_cat, d) ou None
            4) Junta os tensores no formato: [CLS] + num + cat.     (B, L, d), L = n_num_total + n_cat + 1
            6) Aplica dropout nos tokens.
            7) Passa pelo TransformerEncoder.   (B, L, d)
            8) Projeta com a cabeça para 1 logit por exemplo.   (B,)
        Retorno:
            * logits: torch.Tensor -> os logits de classificação de cada exemplo.   (B,)
        """
        B: int = x_num.size(0)

        if x_txt is not None and x_txt.numel() > 0:
            x_num = torch.cat([x_num, x_txt], dim=1)

        t_num: torch.Tensor = self._num_to_tokens(x_num)
        t_cat: torch.Tensor = self._cat_to_tokens(x_cat_list)

        cls: torch.Tensor = self.cls_token.expand(B, 1, self.d)
        if t_cat is None:
            seq: torch.Tensor = torch.cat([cls, t_num], dim=1)
        else:
            seq = torch.cat([cls, t_num, t_cat], dim=1)

        seq = self.drop_tokens(seq)
        enc: torch.Tensor = self.encoder(seq)          
        cls_out: torch.Tensor = enc[:, 0, :]          
        logits: torch.Tensor = self.head(cls_out).squeeze(-1)
        return logits
