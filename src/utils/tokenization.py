import os
from src.utils.seed import seed
from transformers import AutoTokenizer

seed()

PATH = "/home/olavo-dalberto/gpt_ed_assistant/experiments/models"
def load_tokenizer(path: str, use_fast: bool = False) -> None:
    return AutoTokenizer.from_pretrained(path, use_fast=use_fast)

    
def save_tokenizer(tokenizer, path : str=PATH) -> None:
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    tokenizer.save_pretrained(PATH)
    print(f"Tokenizer salvo em: {PATH}")