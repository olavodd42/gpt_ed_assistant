from copy import deepcopy
from ast import literal_eval
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# colunas mínimas para qualquer tarefa supervisionada + grupos
REQUIRED_COLS = ["lab_group_idx", "outcome_critical", "outcome_ed_los"]

def _to_int(x: Any) -> Optional[int]:
    """Converte para int com segurança; retorna None se não der."""
    try:
        if pd.isna(x):
            return None
        # aceita np.int*, str numérica, etc.
        return int(x)
    except Exception:
        return None

def normalize_groups(s: Any) -> List[int]:
    """
    Converte 'lab_group_idx' para uma lista de ints.
    Aceita lista/tupla, string de lista (ex: "[0, 5]"), int único, ou NaN.
    """
    # lista/tupla já vem ok
    if isinstance(s, (list, tuple)):
        out = [_to_int(v) for v in s]
        return [v for v in out if v is not None]

    # string -> tenta literal_eval
    if isinstance(s, str):
        try:
            v = literal_eval(s)
        except Exception:
            v = []
        if isinstance(v, (list, tuple)):
            out = [_to_int(x) for x in v]
            return [x for x in out if x is not None]
        # se a string era um número único
        v_int = _to_int(v)
        return [v_int] if v_int is not None else []

    # número único
    v_int = _to_int(s)
    return [v_int] if v_int is not None else []

def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Checa colunas obrigatórias e levanta erro com lista legível se faltar.
    - Normaliza 'lab_group_idx' para List[int]
    - Converte outcomes para int64 (tratando NaN como erro)
    """
    # 1) checagem de colunas
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam as seguintes colunas obrigatórias no dataset: {missing}")

    # 2) cópia para não mutar o df original
    df = deepcopy(df)

    # 3) normaliza grupos
    df["lab_group_idx"] = df["lab_group_idx"].map(normalize_groups)

    # 4) outcomes -> int64, garantindo valores válidos
    for col in ["outcome_critical", "outcome_ed_los"]:
        if df[col].isna().any():
            na_rows = df.index[df[col].isna()].tolist()
            raise ValueError(f"Coluna '{col}' contém NaN nas linhas: {na_rows[:10]}{'...' if len(na_rows)>10 else ''}")
        try:
            df[col] = df[col].apply(_to_int).astype("int64")
        except Exception as e:
            raise ValueError(f"Falha ao converter '{col}' para inteiro: {e}")

    # 5) (opcional) filtrar IDs de grupo inválidos (ex.: fora de 0..11)
    #    Se preferir, troque por raise em vez de filtrar.
    def _clip_groups(lst: List[int]) -> List[int]:
        return [g for g in lst if (g is not None and 0 <= g <= 11)]

    df["lab_group_idx"] = df["lab_group_idx"].apply(_clip_groups)

    return df
