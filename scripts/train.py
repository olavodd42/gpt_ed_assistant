from src.utils.seed import seed
seed()
import os
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from src.trainer.ModelTrainer import ModelTrainer
from src.utils.tokenization import load_tokenizer
from src.load_data.pickle_load import PickleDataset
from src.load_data.collate import collate_ed


def cleanup():
    gc.collect()                     # força o garbage collector do Python
    torch.cuda.empty_cache()         # limpa cache de memória que não está em uso
    torch.cuda.ipc_collect()         # limpa memória compartilhada entre processos (às vezes útil)

cpkt = "microsoft/BioGPT"
tokenizer = load_tokenizer(cpkt)

train_dataset = PickleDataset("/home/olavo-dalberto/gpt_ed_assistant/data/text/train__outcome_critical.pkl")

print("n amostras:", len(train_dataset))
print(train_dataset[0].keys(), {k: (v if v is None else (v.shape if hasattr(v, "shape") else type(v))) for k,v in train_dataset[0].items()})
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_ed
)


trainer = ModelTrainer(
    cpkt=cpkt,
    tokenizer=tokenizer,
    clf_loss=nn.CrossEntropyLoss(),
    sel_loss=nn.CrossEntropyLoss(),
    baseline=False,
    use_amp=True,
    grad_clip=1.,
    accumulation_steps=16
)

trainer.configure_optimizer(optim.AdamW, lr=2e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.opt, T_max=10)

batch = next(iter(train_loader))
ids = batch["input_ids"]
eos_id = trainer.model.cfg.eos_token_id
pad_id = trainer.model.cfg.pad_token_id

# def debug_eos(dataset, tokenizer, n=5):
#     for i in range(n):
#         ex = dataset[i]
#         tokens = ex["input_ids"]
#         text = tokenizer.decode(tokens, skip_special_tokens=False)
#         print("==== Sample", i, "====")
#         print(text)
#         print("EOS positions:", [j for j,t in enumerate(tokens) if t == tokenizer.eos_token_id])
# debug_eos(train_dataset, tokenizer=tokenizer)



for epoch in range(1):
    if torch.cuda.is_available():
        print(f"[INFO] GPU memory before cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        cleanup()
        print(f"[INFO] GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    logs = trainer.train_epoch(train_loader, alpha=1.0)
    print(f"epoch {epoch}: {logs}")
