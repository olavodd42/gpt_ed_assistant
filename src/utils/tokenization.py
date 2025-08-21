import os
from src.utils.seed import seed
from transformers import AutoTokenizer

seed()

PATH = 'models'
class Tokenizer:
    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def save_tokenizer(self, path : str=PATH) -> None:
        if not os.path.isdir(PATH):
            os.makedirs(PATH)

        self.tokenizer.save_pretrained(PATH)