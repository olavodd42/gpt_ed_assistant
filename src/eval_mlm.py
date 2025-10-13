from pathlib import Path
import os, re, json
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import math
ROOT = Path("/mnt/dados/gpt_ed_assistant/models/bioclinicalbert_mimicnote").resolve()

def last_ckpt(root: Path) -> Path | None:
    cks = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not cks: return None
    ck = max(cks, key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    return ck

def has_model_files(p: Path) -> bool:
    return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()

def ensure_tokenizer_files(p: Path):
    needed = ["tokenizer.json","vocab.txt","special_tokens_map.json","tokenizer_config.json"]
    missing = [f for f in needed if not (p / f).exists()]
    if missing:
        # copia do tokenizer base
        base_tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
        base_tok.save_pretrained(p)

# 1) Tente carregar do último checkpoint (geralmente onde está o model.safetensors)
ckpt = last_ckpt(ROOT)
load_dir = ckpt if (ckpt and has_model_files(ckpt)) else ROOT
if not has_model_files(load_dir):
    raise FileNotFoundError(f"Nenhum arquivo de modelo encontrado em: {load_dir}")

# 2) Garanta arquivos do tokenizer (se faltar, preenche)
ensure_tokenizer_files(load_dir)

# 3) Carregar modelo/tokenizer (suporta .safetensors)
tok = AutoTokenizer.from_pretrained(str(load_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(load_dir))
model.tie_weights()  # evita aviso do decoder

# 4) (Opcional) Consolidar na raiz para uso futuro (um diretório único)
#    Isso copia o estado do último checkpoint para ROOT
if load_dir != ROOT:
    model.save_pretrained(str(ROOT))
    tok.save_pretrained(str(ROOT))

# 5) Teste rápido (fill-mask)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tok)
print(fill_mask("The patient was admitted to the [MASK] unit due to respiratory failure.")[0])
print(fill_mask("He was given [MASK] for chest pain.")[0])
print(fill_mask("Her blood pressure was [MASK] mmHg on arrival.")[0])

print(fill_mask("The [MASK] showed bilateral infiltrates.")[0])
print(fill_mask("He was started on [MASK] for pneumonia.")[0])
print(fill_mask("Her sodium level was [MASK] mEq/L.")[0])
