# src/load_data/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence

def _to_long_tensor(x):
    if torch.is_tensor(x):
        return x.to(dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long)

def collate_ed(batch):
    """
    Espera itens do PickleDataset com chaves:
      - input_ids: 1D tensor/list de tamanho T
      - attention_mask: 1D tensor/list de tamanho T
      - label: escalar (int ou tensor 0-D)
      - action_group_ids: list[int] (uma por <eos> interno)
    Retorna:
      - input_ids: [B,T]
      - attention_mask: [B,T]
      - label: [B]
      - action_group_ids: list[int] (achatada no batch)
    """
    # --- input_ids / attention_mask ---
    ids_list  = [_to_long_tensor(b["input_ids"])     for b in batch]   # cada um: [T] (ou [<=T])
    attn_list = [_to_long_tensor(b["attention_mask"]) for b in batch]  # cada um: [T]

    # Se por algum motivo os comprimentos variarem, fazemos pad aqui.
    # Se você tem comprimento fixo (ex.: 384), isso vira apenas um stack.
    same_len = len({x.numel() for x in ids_list}) == 1
    if same_len:
        input_ids     = torch.stack(ids_list,  dim=0)   # [B,T]
        attention_mask= torch.stack(attn_list, dim=0)   # [B,T]
    else:
        # use padding (PAD=0 para ids, 0 na mask)
        input_ids      = pad_sequence(ids_list,  batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)

    # --- labels ---
    # Cada label pode vir como int, np.int, tensor 0-D, etc.
    labels = []
    for b in batch:
        y = b["label"]
        if torch.is_tensor(y):
            # tensor 0-D -> escalar
            y = int(y.item())
        else:
            y = int(y)
        labels.append(y)
    labels = torch.as_tensor(labels, dtype=torch.long)  # [B]

    # --- actions (achatado) ---
    actions = []
    for b in batch:
        ag = b.get("action_group_ids")
        if ag is None:
            continue
        # garantir inteiros
        actions.extend([int(x) for x in ag])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "action_group_ids": actions
    }
