import math
import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer)
from datasets import load_from_disk

DATA_DIR   = "data/ds_mlm_bioclinicalbert"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"      # ou seu modelo DAPT local
tok   = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
coll  = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)
ds = load_from_disk(DATA_DIR)

args = TrainingArguments(
    output_dir="tmp_eval",
    report_to="none",
    # ↓↓↓ cortes de memória na 4GB ↓↓↓
    per_device_eval_batch_size=4,      # TENTE 4; se OOM, use 2
    fp16=True,                         # ajuda bastante em RTX
    bf16=False,
    dataloader_num_workers=0,          # evita picos de RAM/VRAM
    eval_accumulation_steps=64,        # reduz pico durante o eval
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL,
    # offload (se ainda der OOM):
    # device_map="auto", offload_folder="offload"   # descomente se precisar
)

trainer = Trainer(model=model, args=args, data_collator=coll, eval_dataset=ds["test"])

torch.cuda.empty_cache()

metrics = trainer.evaluate()
print("Loss:", metrics["eval_loss"], "PPL:", math.exp(metrics["eval_loss"]))
