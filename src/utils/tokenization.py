import os
from src.utils.seed import seed
from transformers import AutoTokenizer

seed()

PATH = "/home/olavo-dalberto/gpt_ed_assistant/experiments/models"
def load_tokenizer(path: str, use_fast: bool = False) -> None:
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast)
    changed = False

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        changed = True
    if tokenizer.pad_token is None:
        # para modelos sem pad, reaproveita eos como pad
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        changed = True

    if changed:
        tokenizer.save_pretrained("/mnt/dados/gpt_ed_assistant/.cache/tokenizer_biogpt_custom")
        tokenizer = AutoTokenizer.from_pretrained("/mnt/dados/gpt_ed_assistant/.cache/tokenizer_biogpt_custom")

    return tokenizer

    
def save_tokenizer(tokenizer, path : str=PATH) -> None:
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    tokenizer.save_pretrained(PATH)
    print(f"Tokenizer salvo em: {PATH}")