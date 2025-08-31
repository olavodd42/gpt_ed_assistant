from transformers import AutoTokenizer
from typing import Dict, Any, List

def load_tokenizer(cpkt: str) -> AutoTokenizer:
    """
    Carrega o tokenizer do modelo.
    Parâmetros:
        - cpkt: str -> caminho do tokenizer.
    Retorna:
        - tokenizer: transformers.AutoTokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(cpkt)

    if tokenizer.eos_token is None: tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_texts(tokenizer: AutoTokenizer, texts: List[str], max_len: int = 256) -> Dict[str, Any]:
    """
    Tokeniza o texto passado.
    Parâmetros:
        - tokenizer: transformers.AutoTokenizer.
        - texts: List[str] -> lista de palavras do texto.
        - max_len: int -> máximo comprimento para cada segmento de texto utilizado pelo tokenizer.
    Retorna:
        - Dict[str, Any] -> dicionário contendo input_ids e attention_mas gerados pelo tokenizer.
    """

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=False
    )

    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}