# scripts/build_corpus_from_notes_streaming.py
import re, hashlib, os, gc
import pandas as pd
from pathlib import Path

OUT = Path("data/corpus_mlm.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Limpeza / regex
# ---------------------------
HEADER_PATTS = [
    r"^name:\s*.*", r"^unit\s*no:\s*.*", r"^admission\s*date:\s*.*", r"^discharge\s*date:\s*.*",
    r"^date\s*of\s*birth:\s*.*", r"^service:\s*.*", r"^attending:\s*.*", r"^allergies:\s*.*",
]
SECTION_TITLES = [
    "Chief Complaint:", "History of Present Illness:", "Past Medical History:",
    "Social History:", "Family History:", "Physical Exam:", "Pertinent Results:",
    "Brief Hospital Course:", "Medications on Admission:", "Discharge Medications:",
    "Discharge Disposition:", "Discharge Diagnosis:", "Discharge Condition:",
    "Discharge Instructions:", "Followup Instructions:", "EXAMINATION:", "TECHNIQUE:",
    "COMPARISON:", "FINDINGS:", "IMPRESSION:"
]
MULTI_UNDERS = re.compile(r"_+")

def clean_discharge_text(t: str) -> str:
    t = t.replace("\r\n","\n").replace("\r","\n")
    lines = []
    for raw in t.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if any(re.match(p, s, flags=re.I) for p in HEADER_PATTS):
            continue
        s = MULTI_UNDERS.sub(" ", s)
        lines.append(s)
    txt = "\n".join(lines)
    for sec in SECTION_TITLES:
        txt = re.sub(rf"\n?{re.escape(sec)}\s*", f"\n{sec}\n", txt, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", txt).strip()

def clean_radiology_text(t: str) -> str:
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = MULTI_UNDERS.sub(" ", t)
    for sec in ["EXAMINATION:", "INDICATION:", "TECHNIQUE:", "COMPARISON:", "FINDINGS:", "IMPRESSION:"]:
        t = re.sub(rf"\n?{sec}\s*", f"\n{sec}\n", t, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def clean_ed_text(t: str) -> str:
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = MULTI_UNDERS.sub(" ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def clean_row(cat: str, text: str) -> str:
    tx = str(text or "")
    if cat == "discharge":
        return clean_discharge_text(tx)
    elif cat == "radiology":
        return clean_radiology_text(tx)
    else:
        return clean_ed_text(tx)

# ---------------------------
# Writers
# ---------------------------
def append_to_csv(df: pd.DataFrame, header_written: bool) -> bool:
    df.to_csv(OUT, mode="a", index=False, header=not header_written)
    return True

# ---------------------------
# Streamers
# ---------------------------
def stream_discharge(fp: str, chunksize: int = 100_000, encoding: str = "utf-8"):
    usecols = ["note_id","subject_id","hadm_id","note_type","note_seq","charttime","storetime","text"]
    reader = pd.read_csv(
        fp, chunksize=chunksize, usecols=usecols, dtype=str,
        on_bad_lines="skip", engine="python", encoding=encoding
    )
    for chunk in reader:
        chunk.columns = [c.lower() for c in chunk.columns]
        out = pd.DataFrame({
            "subject_id": chunk["subject_id"],
            "hadm_id":    chunk["hadm_id"],
            "note_id":    chunk["note_id"],
            "category":   "discharge",
            "charttime":  chunk["charttime"],
            "text":       chunk["text"].astype(str).map(lambda s: s.strip())
        })
        # limpeza
        out["text"] = out["text"].map(lambda s: clean_row("discharge", s))
        # filtro comprimento
        out = out[out["text"].str.len() >= 50]
        yield out

def stream_radiology(fp: str, chunksize: int = 100_000, encoding: str = "utf-8", cap_rows: int = 300_000):
    # tenta descobrir header e colunas
    head = pd.read_csv(fp, nrows=1, dtype=str, on_bad_lines="skip", engine="python", encoding=encoding)
    head.columns = [c.lower() for c in head.columns]
    colmap = {"subject_id": None, "hadm_id": None, "note_id": None, "charttime": None, "text": None}
    for k in list(colmap.keys()):
        if k in head.columns:
            colmap[k] = k
        else:
            # tenta achar equivalente (case-insensitive)
            for c in head.columns:
                if c.lower() == k:
                    colmap[k] = c
                    break
    usecols = [v for v in colmap.values() if v is not None]
    reader = pd.read_csv(
        fp, chunksize=chunksize, usecols=usecols if usecols else None,
        dtype=str, on_bad_lines="skip", engine="python", encoding=encoding
    )
    written = 0
    for chunk in reader:
        chunk.columns = [c.lower() for c in chunk.columns]
        # renomeia para schema final
        for k, v in colmap.items():
            if v is None:
                chunk[k] = pd.NA
            elif v != k:
                chunk.rename(columns={v: k}, inplace=True)
        out = pd.DataFrame({
            "subject_id": chunk.get("subject_id", pd.NA),
            "hadm_id":    chunk.get("hadm_id", pd.NA),
            "note_id":    chunk.get("note_id", pd.NA),
            "category":   "radiology",
            "charttime":  chunk.get("charttime", pd.NA),
            "text":       chunk.get("text", "").astype(str).map(lambda s: s.strip())
        })
        out = out[out["text"].str.len() > 0]
        # limpeza
        out["text"] = out["text"].map(lambda s: clean_row("radiology", s))
        out = out[out["text"].str.len() >= 50]
        # CAP de linhas totais de radiology para não esmagar o corpus
        if cap_rows is not None and written + len(out) > cap_rows:
            out = out.iloc[: max(0, cap_rows - written)]
        written += len(out)
        yield out
        if cap_rows is not None and written >= cap_rows:
            break

def build_ed_cc(fp: str, sample_frac: float = 0.2, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(fp, dtype=str, low_memory=False, encoding=encoding)
    df.columns = [c.lower() for c in df.columns]
    if "chiefcomplaint" not in df.columns:
        raise ValueError("master.csv precisa ter coluna 'chiefcomplaint'")

    ed = df[~df["chiefcomplaint"].isna()].copy()
    ed = ed[ed["chiefcomplaint"].str.len() > 2].copy()

    if "subject_id" not in ed.columns: ed["subject_id"] = "-1"
    if "hadm_id"   not in ed.columns: ed["hadm_id"]   = "-1"

    def norm_cc(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip())
        # Template maior pra evitar corte por tamanho
        return f"Chief Complaint:\n{s}\nContext: Emergency triage descriptor."
    ed["text"] = ed["chiefcomplaint"].map(norm_cc)
    ed["category"] = "ed_cc"
    ed["note_id"] = "-1"
    ed["charttime"] = pd.NaT

    ed = ed[["subject_id","hadm_id","note_id","category","charttime","text"]].copy()
    # dedup por texto
    ed["txt_hash"] = ed["text"].map(lambda x: hashlib.md5(("ed_cc||"+x).encode()).hexdigest())
    ed = ed.drop_duplicates(subset=["txt_hash"]).drop(columns=["txt_hash"])

    # amostra uma fração (20%) — ajuste se quiser mais/menos
    if 0 < sample_frac < 1.0:
        ed = ed.sample(frac=sample_frac, random_state=42)

    # não aplicar corte de 50 chars em ed_cc (são curtas mesmo)
    ed = ed[ed["text"].str.len() >= 10]
    return ed

# ---------------------------
# Main streaming
# ---------------------------
def main():
    discharge_csv = "data/raw/discharge.csv"
    radiology_csv = "data/raw/radiology.csv"
    mimic_ed_csv  = "data/raw/master.csv"

    # remove CSV anterior (se existir)
    if OUT.exists():
        OUT.unlink()

    header_written = False

    # DISCHARGE (stream)
    for i, chunk in enumerate(stream_discharge(discharge_csv, chunksize=80_000)):
        if not chunk.empty:
            append_to_csv(chunk, header_written)
            header_written = True
        del chunk
        gc.collect()

    # RADIOLOGY (stream + cap de linhas)
    for i, chunk in enumerate(stream_radiology(radiology_csv, chunksize=80_000, cap_rows=300_000)):
        if not chunk.empty:
            append_to_csv(chunk, header_written)
            header_written = True
        del chunk
        gc.collect()

    # ED chief complaint (pequeno, sem streaming)
    try:
        ed = build_ed_cc(mimic_ed_csv, sample_frac=0.2)
        if not ed.empty:
            append_to_csv(ed, header_written)
            header_written = True
    except Exception as e:
        print("AVISO: ED chiefcomplaint não incluído:", e)

    # Relatório rápido
    dfh = pd.read_csv(OUT, usecols=["category"])
    print("Corpus salvo:", OUT, "linhas:", len(dfh))
    print(dfh["category"].value_counts())

if __name__ == "__main__":
    # dicas de performance
    pd.options.mode.copy_on_write = True
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
