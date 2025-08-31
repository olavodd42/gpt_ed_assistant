from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

class ClinicalDataset:
    def __init__(self, enc: Dict[str, torch.Tensor], labels: List[int], lab_groups: List[List[int]]) -> None:
        self.input_ids = enc["input_ids"]   # [N, T]
        self.attention_mask = enc["attention_mask"] # [N, T]
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.lab_groups = lab_groups    # List[List[int]]

    def __len__(self): return self.input_ids.size(0)

    def __getitem__(self, i: int) -> Dict:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            label=self.labels[i],
            action_group_ids=self.lab_groups[i]
        )
    
def collate_ed(batch: List[Dict[str, Any]], eos_id: int, pad_id: int) -> Dict:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    # achata as ações
    actions_groups = [int(x) for b in batch for x in b.get("action_group_ids",[])]
    
    # Verifica tokens [EOS]
    eos_mask = (input_ids == eos_id)
    pad_mask = eos_mask.int()
    last = pad_mask.argmax(dim=1) - 1
    no_pad = pad_mask.sum(dim=1) == 0
    B, T = input_ids.size()
    last[no_pad] = T - 1

    # Remove último token [EOS]
    eos_mask[torch.arange(B, device="cuda"), last] = False

    # Remove primeiro token [EOS]
    first = pad_mask.argmax(dim=1)
    eos_mask[torch.arange(B, device="cuda"), first] = False

    n_actions_from_eos = int(eos_mask.sum().item())

    if n_actions_from_eos != len(actions_groups):
        print(f"[WARN] mismatch ações: eos={n_actions_from_eos} vs labels={len(actions_groups)}")

    return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=labels,
            action_group_ids=actions_groups
        )