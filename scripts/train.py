#!/usr/bin/env python3
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import wandb
wandb.init(
    project="mimic-ed-nlp",    
    name="bioclinicalbert_run"
)

# --- 0) paths ---
TR = "/home/olavo-dalberto/gpt_ed_assistant/data/processed/train_linearized.csv"
VA = "/home/olavo-dalberto/gpt_ed_assistant/data/processed/valid_linearized.csv"
TE = "/home/olavo-dalberto/gpt_ed_assistant/data/processed/test_linearized.csv"

TEXT_COL = "text_labs"            # ou "text_full"
TARGET   = "outcome_critical"     # ou "outcome_ed_los"

# --- 1) carregar CSVs e selecionar colunas ---
train_df = pd.read_csv(TR)[[TEXT_COL, TARGET]].rename(columns={TARGET: "labels"})
valid_df = pd.read_csv(VA)[[TEXT_COL, TARGET]].rename(columns={TARGET: "labels"})
test_df  = pd.read_csv(TE)[[TEXT_COL, TARGET]].rename(columns={TARGET: "labels"})

# garantir tipo int para labels
for df in (train_df, valid_df, test_df):
    df["labels"] = df["labels"].astype(int)

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(valid_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})

# --- 2) tokenizer/model ---
model_ckpt = "emilyalsentzer/Bio_ClinicalBERT"  # pode trocar por "microsoft/BioGPT-Large" (exige GPU maior)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, padding="max_length", max_length=256)

ds_tok = ds.map(tokenize, batched=True)
# remover coluna de texto para não confundir o Trainer
ds_tok = ds_tok.remove_columns([TEXT_COL])
# setar formato PyTorch
ds_tok = ds_tok.with_format("torch")

# --- 3) modelo ---
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

# (Opcional) lidar com desbalanceamento calculando pos_weight
# Calcula pos_weight = (neg/pos)
pos = (train_df["labels"] == 1).sum()
neg = (train_df["labels"] == 0).sum()
if pos > 0:
    pos_weight = float(neg / pos)
else:
    pos_weight = 1.0

# Para aplicar pos_weight em BCEWithLogits (só para modelos com essa loss):
# Para BertForSequenceClassification a loss é CrossEntropy por padrão.
# Se quiser usar pos_weight, você pode:
# - trocar para BCEWithLogits num wrapper customizado OU
# - usar class weights no Trainer via data_collator/compute_loss custom.
# Aqui seguimos com CrossEntropy simples para simplicidade.

# --- 4) métricas ---
def compute_metrics(p):
    logits, labels = p
    probs = (logits[:, 1]).astype(np.float32) if isinstance(logits, np.ndarray) else logits[:, 1]
    preds = (probs >= 0.5).astype(int) if isinstance(probs, np.ndarray) else (probs >= 0.5).cpu().numpy().astype(int)
    labels = labels if isinstance(labels, np.ndarray) else labels.cpu().numpy()

    metrics = {}
    # Accuracy / F1 binário
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds)
    # AUC e AUPRC exigem ambos os rótulos presentes; proteger contra casos degenerados
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, (logits[:, 1] if isinstance(logits, np.ndarray) else logits[:, 1].cpu().numpy()))
        metrics["auprc"] = average_precision_score(labels, (logits[:, 1] if isinstance(logits, np.ndarray) else logits[:, 1].cpu().numpy()))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics

# --- 5) treino ---
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    logging_steps=50,
    report_to="wandb",
    run_name="bioclinicalbert_ed"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
print("\n=== VAL ===")
print(trainer.evaluate(ds_tok["validation"]))
print("\n=== TEST ===")
print(trainer.evaluate(ds_tok["test"]))
