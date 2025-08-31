# src/dataio/reader.py
from typing import List, Tuple, Dict
import pandas as pd
from .schema import validate_frame
from src.text.linearize import build_text_example

def read_dataset(
    df: pd.DataFrame,
    task_col: str,
    eos_token: str,
    ehr_cols: List[str],
    group_name: Dict[int, str],
    group_cols: Dict[int, List[str]],
) -> Tuple[List[str], List[int], List[List[int]], List[int]]:
    """
    Lineariza o dataset para treino/inferência.
    Retorna:
      texts:         List[str]
      y:             List[int]
      lab_groups_all:List[List[int]]
      n_actions_all: List[int]
    """
    df = validate_frame(df)  # garante tipos/colunas e normaliza lab_group_idx

    texts: List[str] = []
    y: List[int] = []
    lab_groups_all: List[List[int]] = []
    n_actions_all: List[int] = []

    for _, row in df.iterrows():
        # 'lab_group_idx' já é lista (schema.normalize_groups)
        groups = [int(x) for x in row["lab_group_idx"]]

        txt, n_actions = build_text_example(
            row,
            lab_groups=groups,
            eos_token=eos_token,
            ehr_cols=ehr_cols,
            group_name=group_name,
            group_cols=group_cols,
        )

        texts.append(txt)
        y.append(int(row[task_col]))
        lab_groups_all.append(groups)
        n_actions_all.append(n_actions)

    return texts, y, lab_groups_all, n_actions_all
