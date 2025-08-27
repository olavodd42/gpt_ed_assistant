import os
import pickle
import torch
from torch.utils.data import Dataset

class PickleDataset(Dataset):
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        if os.path.getsize(filepath) == 0:
            raise ValueError(f"O arquivo está vazio (0 bytes): {filepath}")

        # LEITURA CORRETA
        with open(filepath, "rb") as f:
            self.data = pickle.load(f)  # lista de dicts: input_ids, attention_mask, label, lab_groups...

        # checagem básica
        if not isinstance(self.data, list) or len(self.data) == 0:
            raise ValueError(f"Conteúdo inesperado no pickle: {type(self.data)} ou lista vazia")

        # garante chaves mínimas
        for k in ("input_ids", "attention_mask", "label"):
            if k not in self.data[0]:
                raise KeyError(f"Item do pickle não tem a chave obrigatória '{k}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        rec = self.data[idx]
        return {
            "input_ids": torch.tensor(rec["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(rec["attention_mask"], dtype=torch.long),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            # pode ser None se você estiver no baseline/sem selector
            "action_group_ids": rec.get("lab_groups", None),
        }
