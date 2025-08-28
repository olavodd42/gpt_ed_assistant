# src/load_data/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence

def _to_long_tensor(x):
    if torch.is_tensor(x):
        return x.to(dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long)

def collate_ed(batch, tokenizer):
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
    ids_list  = [_to_long_tensor(b["input_ids"])     for b in batch]   # [T]
    attn_list = [_to_long_tensor(b["attention_mask"]) for b in batch]  # [T]

    same_len = len({x.numel() for x in ids_list}) == 1
    if same_len:
        input_ids     = torch.stack(ids_list,  dim=0)   # [B,T]
        attention_mask= torch.stack(attn_list, dim=0)   # [B,T]
    else:
        # usa padding (PAD=0 para ids, 0 na mask)
        input_ids      = pad_sequence(ids_list,  batch_first=True, padding_value=tokenizer.pad_token_id or 0)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)

    # --- labels ---
    # Cada label pode vir como int, np.int, tensor 0-D, etc.
    labels = []
    for b in batch:
        y = b["label"]
        y = int(y.item()) if torch.is_tensor(y) else int(y)
        labels.append(y)
    labels = torch.as_tensor(labels, dtype=torch.long)  # [B]

    # --- actions (achatado) ---
    actions = []
    for b in batch:
        ag = b.get("action_group_ids")
        if ag is None:
            continue
        # garante inteiros
        actions.extend([int(x) for x in ag])

    # --- Checagem de coerência: nº de <eos> internos = nº de ações no batch ---
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    internal_eos_mask = (input_ids == eos_id)
    
    # remove o <eos> final por amostra
    is_pad = (input_ids == pad_id).int()
    no_pad = (is_pad.sum(dim=1) == 0)
    last_idx = is_pad.argmax(dim=1) - 1
    T = input_ids.size(1)
    last_idx[no_pad] = T - 1

    # remove o <eos> final
    internal_eos_mask[torch.arange(input_ids.size(0), device="cuda"), last_idx] = False

    # Remove o <eos> da triagen
    first_eos_idx = internal_eos_mask.int().argmax(1)
    internal_eos_mask[torch.arange(input_ids.size(0)), first_eos_idx] = False

    n_actions_from_eos    = int(internal_eos_mask.sum().item())
    n_actions_from_labels = len(actions)
    if n_actions_from_eos != n_actions_from_labels:
        print(f"[WARN] mismatch ações: eos={n_actions_from_eos} vs labels={n_actions_from_labels}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "action_group_ids": actions
    }
