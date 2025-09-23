import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
from numpy.typing import NDArray

DEVICE: str  = "cuda" if torch.cuda.is_available() else "cpu"

class EDDataset(Dataset):
    """
    Subclasse de torch.utils.data.Dataset que organiza os dados
    processados em formato compatível com o Dataloader do Pytorch.
    Parâmetros:
        * Xnum: NDArray[np.float32] -> matriz contendo dados de colunas numéricas.  (N, F_num)
        * Xcat: Optional[NDArray[np.int64]] -> lista de matrizes contendo dados de colunas categóricas mapeadas para inteiro, se houver.    [(N,),...]
        * Xtxt: Optional[NDArray[np.float32]] -> matriz contendo dados vectorizados da coluna de texto, se houver.  (N, D_txt) ou None
        * y: NDArray[np.number] -> vetor com os valores da variável alvo.   (N,)
    """
    def __init__(self,
                 Xnum: NDArray[np.float32],
                 Xcat: Optional[List[NDArray[np.int64]]],
                 y: NDArray[np.number],
                 *,
                 Xtxt: Optional[NDArray[np.float32]] = None
                 ) -> None:
        super().__init__()
        self.Xnum: NDArray[np.float32] = Xnum   # (N, F_num)
        self.Xcat: List[NDArray[np.int64]] = Xcat or []   # [(N,), ...]
        self.Xtxt: Optional[NDArray[np.float32]] = Xtxt     # (N, D_txt)
        self.y: NDArray[np.number] = y  # (N,)

        N = self.Xnum.shape[0]
        assert all(len(a) == N for a in self.Xcat), "Xcat com comprimentos diferentes"
        if self.Xtxt is not None: assert self.Xtxt.shape[0] == N, "Xtxt N inconsistente"
        assert len(self.y) == N, "y N inconsistente"

    def __len__(self):
        """Retorna o tamanho do dataset (número de pacientes)."""
        return self.Xnum.shape[0]
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Obtém o elemento localizado no índice i no formato esperado pelo Dataloader.
        Parâmetros:
            * i: int -> índice desejado.
        Retorna:
            * item: Dict[str, torch.Tensor | list[torch.Tensor]] -> dict contendo os tensores de cada um tipo de coluna (numérica, categórica, textual e alvo).
        """

        item: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "x_num": torch.from_numpy(self.Xnum[i]).to(torch.float32),
            "x_cat_list": [torch.tensor(a[i], dtype=torch.long) for a in self.Xcat],
            "y": torch.tensor(self.y[i]),
        }

        if self.Xtxt is not None:
            item["x_txt"] = torch.from_numpy(self.Xtxt[i]).to(torch.float32)

        return item
    
def collate_fn(batch: List[Dict[str, torch.Tensor | List[torch.Tensor]]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    """
    Monta o batch adequadamente para o DataLoader, empilhando as variáveis.
    Parâmetors:
        * batch: List[Dict[str, torch.Tensor | List[torch.Tensor]]] -> lista de dicionários no formato:
                ```
                {
                    "x_num": torch.from_numpy(self.Xnum[i]).to(torch.float32),
                    "x_cat_list": [torch.tensor(a[i], dtype=torch.long) for a in self.Xcat],
                    "y": torch.tensor(self.y[i]),
                }
                ```
    Retorna:
        * out: Dict[str, torch.Tensor | List[torch.Tensor]] -> dicionário com campos no formato de batch:
            ```
            {
                "x_num": torch.FloatTensor  # (B, F_num),
                "x_cat_list": List[torch.Tensor]    # (cada um (B,)),
                "x_txt": torch.Tensor      # (B, D_txt),   # se existir
                "y": torch.Tensor   # (B,)
            }
            ```"""
    B = len(batch)
    x_num: torch.Tensor = torch.stack([b["x_num"] for b in batch], dim=0)   # (B, F_num)
    y: torch.Tensor = torch.stack([b["y"] for b in batch], dim=0)  # (B,)

     # lista de listas -> lista de tensores (B,)
    T = len(batch[0]["x_cat_list"])
    x_cat_list: List[torch.Tensor] = [torch.stack([b["x_cat_list"][t] for b in batch], dim=0) for t in range(T)]
    out: Dict[str, torch.Tensor | List[torch.Tensor]] = {"x_num": x_num, "x_cat_list": x_cat_list, "y": y}

    if "x_txt" in batch[0]:
        x_txt: torch.Tensor = torch.stack([b["x_txt"] for b in batch], dim=0)
        out["x_txt"] = x_txt

    return out