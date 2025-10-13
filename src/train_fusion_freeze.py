

import os, json, warnings, re
from datetime import timedelta
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import OneCycleLR

# ==================== Configurações ====================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PARQUET: str = "data/master_dataset_split.parquet"
NOTES_CSV: str   = "data/corpus_mlm.csv"
TEXT_ENCODER: str = "/mnt/dados/gpt_ed_assistant/models/bioclinicalbert_mimicnote"
BATCH_SIZE: int   = 4
MAX_EPOCHS: int   = 5
PATIENCE: int     = 2            
MAX_STEPS_PER_EPOCH: Optional[int] = None
TH: float           = 8.0          # threshold ED-LOS 

DROPOUT_TAB: float  = 0.2      # Dropout para ramo tabular
DROPOUT_HEAD: float = 0.3      # Dropout do head

# ===== DEV RUN (ensaio rápido) =====
DEV_RUN: bool = False
N_TRAIN: int = 30000
N_VAL: int   = 6000
N_TEST: int  = 6000

EPOCHS: int   = 5
MAX_LEN: int  = 192
LR_HEAD: float  = 5e-4

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x

# ---- Acumulação de gradiente ----
# Batch efetivo ~= BATCH_SIZE * ACCUM_STEPS
ACCUM_STEPS: int = max(1, 32 // BATCH_SIZE)
print(f"[CFG] ACCUM_STEPS={ACCUM_STEPS} (batch efetivo ≈ {BATCH_SIZE*ACCUM_STEPS})")


# ===== Carrega dados =====
df: pd.DataFrame = pd.read_parquet(DATA_PARQUET)  # Dataset tabular
assert "split" in df.columns, "arquivo precisa ter coluna 'split'"
df["chiefcomplaint"] = df.get("chiefcomplaint", "").fillna("")

# Converte ed_los_hours e cria rótulo de LOS
if "ed_los_hours" in df.columns:
    df["ed_los_hours"] = pd.to_numeric(df["ed_los_hours"], errors="coerce")
    df.loc[df["ed_los_hours"] < 0, "ed_los_hours"] = np.nan
    df[f"outcome_ed_los_ge{TH}h"] = (df["ed_los_hours"] >= TH).astype(int)
    TARGET_COLS: List[str] = ["outcome_critical", f"outcome_ed_los_ge{TH}h"]
else:
    TARGET_COLS = ["outcome_critical"]

# === Variáveis tabulares ===
num_cols: List[str] = [c for c in [
    "anchor_age","age","triage_temperature","triage_heartrate",
    "triage_resprate","triage_o2sat","triage_sbp","triage_dbp","triage_pain"
] if c in df.columns]
cat_cols: List[str] = [c for c in [
    "gender","race","ethnicity","arrival_transport","insurance","triage_acuity"
] if c in df.columns]
cci_cols: List[str] = [c for c in df.columns if c.startswith("cci_")]
eci_cols: List[str] = [c for c in df.columns if c.startswith("eci_")]
bin_cols: List[str] = cci_cols + eci_cols

# Prepara dados tabulares
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in bin_cols:
    df[c] = df[c].fillna(0).astype(int)
df_cat: pd.DataFrame = pd.get_dummies(df[cat_cols].fillna("UNK"), prefix=cat_cols, drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
df_num: pd.DataFrame = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
df_bin: pd.DataFrame = df[bin_cols].copy() if bin_cols else pd.DataFrame(index=df.index)
X_all: pd.DataFrame = pd.concat([df_num, df_cat, df_bin], axis=1).astype(np.float32)

# Preparação das notas
def _keep_impression(txt: str) -> str:
    """ Realiza a preparação das notas"""
    if not isinstance(txt, str):
        return ""
    m = re.search(r"\bIMPRESSION:\s*(.*)", txt, flags=re.I | re.S)
    return (m.group(0) if m else txt)[:800]

notes: pd.DataFrame = pd.read_csv(
    NOTES_CSV,
    usecols=["subject_id","hadm_id","category","charttime","text"],
    dtype={"subject_id":"Int64","hadm_id":"Int64","category":"string","text":"string"},
    low_memory=False
)
notes["charttime"] = pd.to_datetime(notes["charttime"], errors="coerce")
df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
df["hadm_id"]    = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
df["edregtime"]  = pd.to_datetime(df["edregtime"], errors="coerce")

# T1 = T0 + 2
df["_t0"] = df["edregtime"]
df["_t1"] = df["_t0"] + timedelta(hours=2)

# Faz um join dos dataframes por meio de "subject_id" e "hadm_id"
key_cols: List[str] = ["subject_id","hadm_id"]
notes = notes[notes["category"].str.lower().isin(["ed","radiology"])].dropna(subset=["charttime"])
# Filtra notas entre T e T+2
merged: pd.DataFrame = df[key_cols + ["_t0","_t1"]].drop_duplicates().merge(notes, on=key_cols, how="left")
merged = merged[(merged["charttime"] >= merged["_t0"]) & (merged["charttime"] <= merged["_t1"])].copy()

# Filtra as informações mais úteis, agrupando as linhas por "subject_id"
# e "hadm_id", concatenando as notas
if not merged.empty:
    mask_rad: pd.Series[bool] = merged["category"].str.lower() == "radiology"
    merged.loc[mask_rad, "text"] = merged.loc[mask_rad, "text"].map(_keep_impression)
    merged = merged.sort_values(["subject_id","hadm_id","charttime"])
    agg: pd.DataFrame = merged.groupby(key_cols)["text"].apply(
        lambda xs: ("\n---\n".join(s for s in xs.astype(str) if s))[:1200]
    ).reset_index().rename(columns={"text":"_notes_t0_t2h"})
    df = df.merge(agg, on=key_cols, how="left")
else:
    df["_notes_t0_t2h"] = ""

df["text"] = (
    "Chief complaint: " + df["chiefcomplaint"].fillna("") +
    " | Vitals: " +
    "T=" + df["triage_temperature"].astype(str) + " " +
    "HR=" + df["triage_heartrate"].astype(str)    + " " +
    "RR=" + df["triage_resprate"].astype(str)     + " " +
    "SpO2=" + df["triage_o2sat"].astype(str)      + " " +
    "SBP=" + df["triage_sbp"].astype(str)         + " " +
    "DBP=" + df["triage_dbp"].astype(str)         + " " +
    "Pain=" + df["triage_pain"].astype(str) +
    "\n\nNotes (T0–T+2h):\n" + df["_notes_t0_t2h"].fillna("")
).str.slice(0, 1400)
df = df.drop(columns=["_t0","_t1"])

# Targets e sanitização
present_targets: List[str]  = [c for c in TARGET_COLS if c in df.columns]
Y_all: pd.DataFrame = df[present_targets].copy()

# Converte targets para binários (0/1)
for c in present_targets:
    Y_all[c] = Y_all[c].map({True:1, False:0}).fillna(Y_all[c]).astype(str)
    Y_all[c] = Y_all[c].str.extract(r"(\d+)").fillna("0").astype(int).clip(0,1)

# Split por rótulo "split" e criação dos datasets
train_mask: pd.Series[bool] = df["split"] == "train"
val_mask: pd.Series[bool]   = df["split"] == "val"
test_mask: pd.Series[bool]  = df["split"] == "test"
X_train_df, X_val_df, X_test_df = X_all[train_mask].copy(), X_all[val_mask].copy(), X_all[test_mask].copy()
Y_train_df, Y_val_df, Y_test_df = Y_all[train_mask].copy(), Y_all[val_mask].copy(), Y_all[test_mask].copy()
text_train, text_val, text_test = df.loc[train_mask,"text"], df.loc[val_mask,"text"], df.loc[test_mask,"text"]

# Normaliza colunas numéricas com StandardScaler
if num_cols:
    med: pd.Series[np.float32] = X_train_df[num_cols].median(numeric_only=True)
    X_train_df[num_cols] = X_train_df[num_cols].fillna(med).astype("float32")
    X_val_df[num_cols]   = X_val_df[num_cols].fillna(med).astype("float32")
    X_test_df[num_cols]  = X_test_df[num_cols].fillna(med).astype("float32")
    scaler: StandardScaler = StandardScaler().fit(X_train_df[num_cols])
    for split_df in (X_train_df, X_val_df, X_test_df):
        split_df[num_cols] = scaler.transform(split_df[num_cols]).astype("float32")

def clean_np(a: npt.NDArray[np.number]) -> npt.NDArray[np.float32]:
    """Limpa os arrays, convertendo-os para arrays numpy"""
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return a

X_train: npt.NDArray[np.float32] = clean_np(X_train_df.to_numpy())
X_val: npt.NDArray[np.float32]   = clean_np(X_val_df.to_numpy())
X_test: npt.NDArray[np.float32]  = clean_np(X_test_df.to_numpy())
Y_train: npt.NDArray[np.float32] = clean_np(Y_train_df.to_numpy())
Y_val: npt.NDArray[np.float32]   = clean_np(Y_val_df.to_numpy())
Y_test: npt.NDArray[np.float32]  = clean_np(Y_test_df.to_numpy())
N_TAB: int = X_train.shape[1]
K: int     = Y_train.shape[1]

# ===== Dataset & DataLoader =====
class EDFusionDataset(Dataset):
    """Cria um Dataset customizado pytorch para o Dataloader."""
    def __init__(self, texts: pd.Series[bool], X: pd.DataFrame, Y: pd.DataFrame):
        self.texts: List[str] = list(texts)
        self.X: pd.DataFrame = X
        self.Y: pd.DataFrame = Y
    def __len__(self) -> int:
        """Retorna o comprimento do dataset."""
        return len(self.texts)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Obtém o elemento do dataset na posição idx."""
        return {
            "text": self.texts[idx] or "",
            "tab": torch.tensor(self.X[idx], dtype=torch.float32),
            "labels": torch.tensor(self.Y[idx], dtype=torch.float32),
        }

tok  = AutoTokenizer.from_pretrained(TEXT_ENCODER, use_fast=True)
bert = AutoModel.from_pretrained(TEXT_ENCODER)
for p in bert.parameters(): p.requires_grad = False  # congela o encoder
bert.gradient_checkpointing_enable() 

train_ds: EDFusionDataset = EDFusionDataset(text_train, X_train, Y_train)
val_ds: EDFusionDataset   = EDFusionDataset(text_val,   X_val,   Y_val)
test_ds: EDFusionDataset  = EDFusionDataset(text_test,  X_test,  Y_test)

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Cria função de collate personalizada para o Dataset."""
    texts: List[str] = [b["text"] for b in batch]
    tabs: torch.Tensor  = torch.stack([b["tab"] for b in batch])
    labs: torch.Tensor  = torch.stack([b["labels"] for b in batch])
    enc: Dict[str, torch.Tensor] = tok(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "tab": tabs,
        "labels": labs
    }

# Cria os DataLoaders
common_loader_kwargs = dict(
    num_workers=4,              
    pin_memory=True,
    persistent_workers=True
)

train_loader: DataLoader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_batch, **common_loader_kwargs
)
val_loader: DataLoader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_batch, **common_loader_kwargs
)
test_loader: DataLoader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_batch, **common_loader_kwargs
)

# ===== Modelo de fusão =====
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Condensa o tensor de shape [batch_size, seq_len, hiddens_size] em um
    tensor de shape [batch_size, hidden_size] pela média dos embeddings de
    todos os tokens válidos.
    """
    mask: torch.Tensor = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

class FusionModel(nn.Module):
    """
    Modelo de fusão texto + tabular.

    - Texto: pega o last_hidden_state do encoder (BERT-like) e faz mean-pooling
             mascarado pelos tokens válidos -> vetor [B, H], depois normaliza (LayerNorm).
    - Tabular: MLP raso com BatchNorm -> vetor [B, 256].
    - Fusão: concatena [texto(H) || tab(256)] e manda para um head MLP -> logits [B, K].

    Obs.: Se quiser ativar fusão com 'gate' (porta), defina use_gate=True abaixo.
    """
    def __init__(self, bert_model: nn.Module, n_tab: int, n_labels: int,
                 use_gate: bool = False):
        super().__init__()
        self.bert = bert_model
        # pega tamanho oculto do encoder
        h: int = getattr(self.bert.config, "hidden_size", 768)

        # ramo tabular: Linear -> Norm -> ReLU -> Dropout
        self.tab: nn.Sequential = nn.Sequential(
            nn.Linear(n_tab, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_TAB),
        )

        # normalização leve do embedding textual
        self.text_norm: nn.LayerNorm = nn.LayerNorm(h)

        self.use_gate: bool = use_gate
        if use_gate:
            # projeta tabular para o mesmo espaço do texto (H) e cria um "gate"
            # gate: Linear -> Sigmoid
            self.tab_to_h: nn.Linear = nn.Linear(256, h)
            self.gate: nn.Sequential = nn.Sequential(
                nn.Linear(h + 256, 1),
                nn.Sigmoid()
            )
            # head: recebe [fused(H) || tab(256)]: Linear -> ReLU -> Dropout -> Linear
            self.classifier: nn.Sequential = nn.Sequential(
                nn.Linear(h + 256, 512),
                nn.ReLU(),
                nn.Dropout(DROPOUT_HEAD),
                nn.Linear(512, n_labels),
            )
        else:
            # head padrão (concat direto: [texto(H) || tab(256)]): Linear -> ReLU -> Dropout -> Linear
            self.classifier = nn.Sequential(
                nn.Linear(h + 256, 512),
                nn.ReLU(),
                nn.Dropout(DROPOUT_HEAD),
                nn.Linear(512, n_labels),
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """Realiza um passo forward do modelo"""

        # --- TEXTO ---
        # last_hidden: [B, T, H]
        last_hidden: torch.Tensor = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # mean-pooling mascarado -> [B, H]
        x_text: torch.Tensor = mean_pool(last_hidden, attention_mask)
        x_text = self.text_norm(x_text)

        # --- TABULAR ---
        x_tab: torch.Tensor = self.tab(tab)            # [B, 256]

        # --- FUSÃO ---
        if self.use_gate:
            # porta g aprende quanta ênfase dar ao texto vs. tabular
            g: torch.Tensor = self.gate(torch.cat([x_text, x_tab], dim=1))   # [B, 1]
            x_tab_h: torch.Tensor = self.tab_to_h(x_tab)                     # [B, H]
            x_fused: torch.Tensor = g * x_text + (1.0 - g) * x_tab_h         # [B, H]
            z: torch.Tensor = torch.cat([x_fused, x_tab], dim=1)             # [B, H+256]
        else:
            # concat simples (rápido e já funciona bem)
            z = torch.cat([x_text, x_tab], dim=1)              # [B, H+256]

        # --- LOGITS ---
        return self.classifier(z)  # [B, n_labels]



model: FusionModel = FusionModel(bert, n_tab=N_TAB, n_labels=K, use_gate=True).to(DEVICE)
# Cálculo de pos_weight por label
pos_weight_vec: List[float] = []
for j in range(K):
    pos = float(Y_train[:, j].sum())
    neg = float(len(Y_train) - pos)
    w   = neg / max(pos, 1.0)
    pos_weight_vec.append(min(w, 1000.0))
pos_weight: torch.Tensor = torch.tensor(pos_weight_vec, dtype=torch.float32, device=DEVICE)
criterion: nn.BCEWithLogitsLoss  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt: AdamW = AdamW(model.parameters(), lr=LR_HEAD)

# ===== Funções de métricas =====
def per_label_metrics(y_true: npt.NDArray[np.int64], y_prob: npt.NDArray[np.float32], names: List[str]) -> Tuple[List[Tuple[str, float, float, float]], float, float]:
    """Calcula as métricas para cada label."""
    out, aps, f1s = [], [], []
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    for j, name in enumerate(names):
        # Obtém valores verdadeiro e predito para label j
        tj, pj = y_true[:, j], y_prob[:, j]
        pos = tj.sum()
        neg = len(tj) - pos

        # Caso a coluna tenha só positivos ou negativos retorna np.nan nas métrcias
        if pos == 0 or neg == 0:
            out.append((name, np.nan, np.nan, pos/len(tj)))
            continue

        # Calcula AP (AUCPRC), F1-score em todos treshrolds e obtém o máximo
        ap  = average_precision_score(tj, pj)
        prec, rec, thr = precision_recall_curve(tj, pj)
        f1  = (2 * prec * rec / (prec + rec + 1e-9)).max()
        out.append((name, ap, f1, pos/len(tj)))
        aps.append(ap)
        f1s.append(f1)
    return out, float(np.nanmean(aps) if aps else np.nan), float(np.nanmean(f1s) if f1s else np.nan)

def best_thresholds_by_label(y_true: npt.NDArray[np.int64], y_prob: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Obtém o melhor treshrold para cada label."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    thr_best = []
    for j in range(y_true.shape[1]):
        tj, pj = y_true[:, j], y_prob[:, j]
        if tj.sum() == 0 or tj.sum() == len(tj):
            thr_best.append(0.5); continue
        prec, rec, thr = precision_recall_curve(tj, pj)
        f1 = (2 * prec * rec / (prec + rec + 1e-9))
        idx = int(np.argmax(f1))
        thr_best.append(float(thr[idx - 1]) if len(thr) else 0.5)
    return np.array(thr_best, dtype=np.float32)

# ===== Loop de treino com early stopping & unfreezing =====
scaler_amp: torch.amp.GradScaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
best_val_loss: float = float("inf")
epochs_no_improve: int = 0
history: Dict[str, List[float]] = {"train_loss":[], "val_loss":[], "train_macro_f1":[], "val_macro_f1":[], "train_macro_ap":[], "val_macro_ap":[]}

log_every: int = 500
steps_per_epoch: int = len(train_loader) // max(1, ACCUM_STEPS)
scheduler: OneCycleLR = OneCycleLR(
    opt,
    max_lr=LR_HEAD,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=10
)

# --- scheduler (já está OK) ---

def run_epoch(loader: DataLoader, training: bool=True, return_raw: bool=False) -> Tuple[float, float, float] | Tuple[float, float, float, float, float]:
    """Roda uma época de treino do modelo multimodal, calcula as métricas de treino/eval da época."""
    model.train() if training else model.eval()
    tot_loss: float = 0.0
    seen: int = 0
    y_true_list, y_prob_list = [], []
    pbar = tqdm(loader, disable=False)

    if training:
        opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, start=1):
        # Envia os batchs para CUDA (ou CPU)
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        att = batch["attention_mask"].to(DEVICE, non_blocking=True)
        tab = batch["tab"].to(DEVICE, non_blocking=True)
        y   = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                # Realiza o forward pass
                logits: float = model(ids, att, tab)
                loss: float = criterion(logits, y)
                loss = loss / (ACCUM_STEPS if training else 1)

            # Realiza o backward com scaler
            if training:
                scaler_amp.scale(loss).backward()
                if (step % ACCUM_STEPS == 0) or (step == len(loader)):
                    scaler_amp.step(opt)
                    scaler_amp.update()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)

        # Calcula o loss total da época
        bs: int = y.size(0)
        tot_loss += loss.item() * bs * (ACCUM_STEPS if training else 1)
        seen += bs

        with torch.no_grad():
            y_true_list.append(y.detach().cpu().numpy())
            y_prob_list.append(torch.sigmoid(logits).detach().cpu().numpy())

        if step % log_every == 0:
            pbar.set_description(f"{'TRAIN' if training else 'VAL'} loss={tot_loss/max(seen,1):.4f}")

    # Calcula loss médio, concatena y real e probabilidades estimadas, calcula as métricas
    avg_loss = tot_loss / max(len(loader.dataset), 1)
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    rows, macro_ap, macro_f1 = per_label_metrics(y_true, y_prob, present_targets)

    print("Per-label:")
    for name, ap, f1, prev in rows:
        ap_s = f"{ap:.3f}" if ap==ap else "NA"
        f1_s = f"{f1:.3f}" if f1==f1 else "NA"
        print(f"  {name:24s} AP={ap_s:>6}  F1={f1_s:>6}  prev={prev:6.3f}")
    print(f"Macro AP={macro_ap:.3f} | Macro F1={macro_f1:.3f}")

    if return_raw:
        return avg_loss, macro_f1, macro_ap, y_true, y_prob
    return avg_loss, macro_f1, macro_ap

# ===== Loop de treino com early stopping & unfreezing =====
best_val_loss: float = float("inf")
epochs_no_improve: int = 0
history: Dict[str, List[float]] = {"train_loss":[], "val_loss":[], "train_macro_f1":[], "val_macro_f1":[], "train_macro_ap":[], "val_macro_ap":[]}
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch:02d} / {EPOCHS}")
    # Treina o modelo por uma época e, após, roda a avaliação
    tr_loss, tr_f1, tr_ap, _, _ = run_epoch(train_loader, training=True,  return_raw=True)
    va_loss, va_f1, va_ap, y_val_true, y_val_prob = run_epoch(val_loader,   training=False, return_raw=True)

    history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
    history["train_macro_f1"].append(tr_f1); history["val_macro_f1"].append(va_f1)
    history["train_macro_ap"].append(tr_ap); history["val_macro_ap"].append(va_ap)

    print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} F1 {tr_f1:.3f} AP {tr_ap:.3f} || "
          f"val_loss {va_loss:.4f} F1 {va_f1:.3f} AP {va_ap:.3f}")

    # Verifica se houve melhora para early_stopping
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        epochs_no_improve: int = 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/fusion_freeze.pt")
        np.save("models/val_thresholds.npy", best_thresholds_by_label(y_val_true, y_val_prob))
        print(f"[CKPT] Melhor val_loss {va_loss:.6f} — modelo salvo.")
    else:
        epochs_no_improve += 1
        print(f"[EARLY] Sem melhora ({epochs_no_improve}/{PATIENCE}).")
        if epochs_no_improve >= PATIENCE:
            print("[EARLY] Parando cedo por falta de melhora no val_loss.")
            break
    
    # Descongela as 4 últimas camadas na 1 época
    if epoch == 1:
        print("[UNFREEZE] Descongelando últimas 4 camadas do encoder.")
        for p in model.bert.encoder.layer[-4:].parameters():
            p.requires_grad = True

# salva histórico
np.savez("models/train_history.npz", **history)


# ===== Teste final =====

# Roda uma época de eval
model.load_state_dict(torch.load("models/fusion_freeze.pt", map_location=DEVICE))
thr_val = np.load("models/val_thresholds.npy")
_, _, _, y_true_te, y_prob_te = run_epoch(test_loader, training=False, return_raw=True)  # <<< return_raw=True

macro_ap: float  = average_precision_score(y_true_te, y_prob_te, average="macro")
y_pred_te = (y_prob_te >= thr_val.reshape(1, -1)).astype(int)
macro_f1: float  = f1_score(y_true_te, y_pred_te, average="macro", zero_division=0)
print(f"TEST | Macro F1 {macro_f1:.3f}  Macro AUPRC {macro_ap:.3f}")
print("Labels:", present_targets)
print("Thresholds:", {present_targets[i]: float(thr_val[i]) for i in range(len(present_targets))})

# # Gera gráfico de perdas (pode ser plotado posteriormente)
# try:
#     import matplotlib.pyplot as plt
#     plt.plot(history["train_loss"], label="train_loss")
#     plt.plot(history["val_loss"], label="val_loss")
#     plt.xlabel("Época")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("models/loss_curve.png")
# except Exception:
#     pass
