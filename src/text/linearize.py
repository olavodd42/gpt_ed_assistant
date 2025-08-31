# src/text/linearize.py
from typing import List, Tuple, Dict
import pandas as pd

def linearize_ehr(row: pd.Series, ehr_cols: List[str]) -> str:
    """
    Lineariza colunas EHR no formato:
      "col1: v1; col2: v2; ..."
    Ignora colunas ausentes e valores NaN.
    """
    parts = []
    for col in ehr_cols:
        if (col in row) and pd.notna(row[col]):
            parts.append(f"{col}: {row[col]}")
    return "; ".join(parts)

def linearize_group(row: pd.Series, group_name: str, cols: List[str]) -> str:
    """
    Lineariza as colunas de um grupo laboratorial:
      "group_name: c1: v1 | c2: v2 | ..."
    Ignora colunas ausentes e valores NaN.
    Retorna string vazia se nenhum valor válido existir.
    """
    vals = []
    for col in cols:
        if (col in row) and pd.notna(row[col]):
            vals.append(f"{col}: {row[col]}")
    if not vals:
        return ""
    return f"{group_name}: " + " | ".join(vals)

def build_text_example(
    row: pd.Series,
    lab_groups: List[int],
    eos_token: str,
    ehr_cols: List[str],
    group_name: Dict[int, str],
    group_cols: Dict[int, List[str]],
) -> Tuple[str, int]:
    """
    Constrói o texto:
      <EHR>; ... </s> <grupo1: ...> </s> <grupo2: ...> </s> </s>
    Retorna (texto, n_actions) onde n_actions é o nº de <eos> internos de grupos.
    """
    parts: List[str] = []

    # 1) bloco de triagem/EHR
    ehr_txt = linearize_ehr(row, ehr_cols=ehr_cols)
    if ehr_txt:
        parts.append(ehr_txt)
    parts.append(eos_token)  # <eos> após EHR

    # 2) blocos de laboratório
    n_actions = 0
    for g in lab_groups:
        if g not in group_name or g not in group_cols:
            # grupo desconhecido – pula silenciosamente
            continue
        gtxt = linearize_group(row, group_name[g], group_cols[g])
        if gtxt:
            parts.append(gtxt)
            parts.append(eos_token)  # <eos> separando grupos
            n_actions += 1

    # 3) <eos> final
    parts.append(eos_token)

    return " ".join(parts), n_actions
