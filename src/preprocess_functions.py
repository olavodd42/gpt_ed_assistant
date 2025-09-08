import json, math, pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

SEED = 42

def clamp(df: pd.DataFrame, col: str, lower: float, higher: float) -> pd.Series:
    """
    Corta o intervalo de valores possíveis para uma variável.
    Parâmetros:
        * df: pd.DataFrame -> o dataset que contém as variáveis.
        * col: str -> coluna numérica em que deve ocorrer o recorte.
        * lo: float -> límite inferior do intervalo.
        * hi: float -> limite superior do intervalo.
    Retorno:
        * clipped_data: pd.Series -> coluna com os valores ajustados.
    """
    return df[col].clip(lower=lower, upper=higher)

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte as variáveis booleanas do dataset em int8.
    Parâmetros:
        * df: pd.DataFrame -> dataframe contendo os dados.
    Retorno:
        * df: pd.DataFrame -> dataframe com as variáveis convertidas.
    """
    # bool -> int, strings garantidas como object
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype("int8")

    return df

def lengthened_ed_stay(df: pd.DataFrame) -> pd.DataFrame:
    df["legthened_ed_stay"] = df["ed_los_hours"] > 24.0
    return df

def make_feature_lists(df: pd.DataFrame,
                       target_col: str,
                       text_col: Optional[str],
                       extra_cat: Optional[List[str]],
                       optional_cat: Optional[List[str]],
                       drop_misc: Optional[List[str]],
                       ) -> Tuple[List[str], List[str], Optional[str], List[str]]:
    """
    Separa as colunas do DataFrame em:
      * num_cols: numéricas que vão para o MLP tabular
      * cat_cols: categóricas que virarão embeddings
      * text_col_real: coluna de texto (ou None)
      * drops: colunas para descartar (IDs, timestamps crus, outcomes não-alvo)

    Regras chave:
      * Remove qualquer outcome_ que não seja o target (anti-vazamento).
      * Remove IDs e datas cruas (usamos apenas derivados em horas/dias).
      * 'triage_acuity' é tratada como categórica mesmo se vier numérica.
    Parâmetros:
        * df: pd.DataFrame -> o dataframe utilizado,
        * target_col: str -> a variável alvo,
        * text_col: Optional[str] -> variável de texto,
        * extra_cat: List[str ]-> outras variáveis categóricas,
        * optional_cat: List[str] -> variáveis adicionais opcionais,
        * drop_misc: List[str] -> variáveis que não irão ser utilizadas no treino.
    Retorna:
        * num_cols: List[str] -> variáveis numéricas no dataset,
        * cat_cols: List[str] -> variáveis categóricas no dataset,
        * text_col_real: Optional[str] -> variável de texto,
        * drop_outcomes + drop_misc: List[str] -> colunas não utilizadas.
    """
    
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' não encontrado nas colunas do DataFrame.")

    drop_outcomes: List[str] = [c for c in df.columns if c.startswith("outcome_") and c != target_col]
    # filtra apenas as que de fato existem no df
    drop_misc = [c for c in drop_misc if c in df.columns]


    # texto:
    text_col_real: Optional[str] = text_col if (text_col is not None and text_col in df.columns) else None

    # categóricas
    cat_cols: List[str] = [c for c in extra_cat if c in df.columns]
    cat_cols += [c for c in optional_cat if c in df.columns]

    # numéricas = todas as demais elegíveis (inclui 0/1)
    blocked: Set[str] = {target_col, *drop_outcomes, *drop_misc}
    if text_col_real is not None:
        blocked.add(text_col_real)
    num_cols: List[str] = [c for c in df.columns
                if c not in blocked and df[c].dtype != "object"]

    return num_cols, cat_cols, text_col_real, (drop_outcomes + drop_misc)

def apply_clinical_rules(df_train: pd.DataFrame,
                         df_valid: pd.DataFrame,
                         df_test: pd.DataFrame,
                         vital_bounds: Dict[str, Tuple[int, int]],
                         log1p_cands: List[str]
                         ) -> Dict[str, Tuple[float,float]]:
    """
    Aplica regras clínicas:
      1) Clamp fixo em vitais (faixas fisiológicas)
      2) Coerência: SBP >= DBP (triagem e _last) -> se violado, zera (NaN) para imputar depois
      3) Para colunas com cauda pesada (contagens e certos exames): clamp por quantis do treino + log1p

    Parâmetros:
        * df_train: pd.Dataframe -> dataset de treino,
        * df_valid: pd.DataFrame -> dataset de validação,
        * df_test: pd.DataFrame -> dataset de teste,
        * vital_bounds: Dict[str, Tuple[int, int]] -> dicionário contendo valores lo e hi para clamp de cada feature,
        * log1p_cands: List[str] -> colunas passíveis de transformação logarítmica.

    Retorna:
      * limits: Dict[str, Tuple[int, int]] -> dict {col: (lo, hi)} com os limites usados no passo (3) para reprodutibilidade.
    """
    # Triagem/vitais: clamp fixo por faixa
    for col, (lo, hi) in vital_bounds.items():
        for df in (df_train, df_valid, df_test):
            if col in df.columns:
                df[col] = clamp(df, col=col, lower=lo, higher=hi)

    # Coerência: sistólica >= diastólica (triagem e _last)
    for sbp, dbp in [("triage_sbp","triage_dbp"), ("ed_sbp_last","ed_dbp_last")]:
        if sbp in df_train.columns and dbp in df_train.columns:
            for df in (df_train, df_valid, df_test):
                bad: pd.Series[bool] = (df[sbp].notna() & df[dbp].notna() & (df[sbp] < df[dbp]))
                df.loc[bad, [sbp, dbp]] = np.nan

    # Cauda pesada: clamp por quantis do treino + log1p
    limits: Dict[str, Tuple[float, float]] = {}
    for col in log1p_cands:
        if col in df_train.columns:
            # Clamp em (0.5%, 99.5%) -> remove outliers
            lo: float = float(df_train[col].quantile(0.005))
            hi: float = float(df_train[col].quantile(0.995))
            limits[col] = (lo, hi)
            for df in (df_train, df_valid, df_test):
                df[col] = clamp(df, col=col, lower=lo, higher=hi)
                df[col] = np.log1p(df[col])

    return limits

def fit_num_scaler(num_df: pd.DataFrame) -> Tuple[StandardScaler, np.ndarray]:
    """
    Normaliza as variáveis numéricas.
    Parâmetros:
        * num_df: pd.DataFrame -> colunas numéricas do dataframe de treino.
    Retorna:
        * scaler: StandardScaler -> modelo de normalizaćão.
        * Xtr: np.ndarray -> array normalizado. (n_amostras, n_features)
    """
    scaler: StandardScaler = StandardScaler(with_mean=True, with_std=True)
    Xtr: np.ndarray = scaler.fit_transform(num_df.values)
    return scaler, Xtr.astype("float32")

def transform_num(scaler: StandardScaler, num_df: pd.DataFrame) -> np.ndarray:
    """
    Aplica um StandardScaler já ajustado (fit) em colunas numéricas de um DataFrame.
    Parâmetros:
        * scaler: StandardScaler -> modelo de scaler treinado.
        * num_df: pd.DataFrame -> colunas numéricas do dataframe utilizado.
    Retorna:
        * np.ndarray -> array com valores normalizados (n_amostras, n_features)
    """
    return scaler.transform(df[num_cols].values).astype("float32")

def fit_cat_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str,int]]:
    """
    Cria um dicionário de mapeamento categoria->id para cada coluna categórica.
    Parâmetros:
        * df_train: pd.DataFrame -> dataframe para treino.
        * cat_cols: List[str] -> colunas categóricas do dataframe.
    Retorna:
        * maps: Dict[str, Dict[str,int]] -> Um dicionário no formato:
            {
            "coluna1": {"catA": 1, "catB": 2, ..., "__NA__": valor},
            "coluna2": {"X": 1, "Y": 2, ...},
            ...
            }
    """
    maps: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        # Converte valores em string, normaliza NaN
        s: pd.Series = df[c].astype("object").fillna("__NA__")
        
        # Lista de categorias únicas, ordenadas (garante reprodutibilidade)
        cats: pd.Series = pd.Series(s.unique()).astype(str).sort_values(kind="mergesort")

        # Cria o dicionário: cada categoria recebe um índice começando em 1
        # Obs: índice 0 fica reservado para "UNK" (Unknown).
        maps[c] = {cat: i+1 for i, cat in enumerate(cats)}

    return maps

def apply_cat_maps(df: pd.DataFrame, cat_cols: List[str], maps: Dict[str,Dict[str,int]]) -> List[np.ndarray]:
    """
    Transforma colunas categóricas do DataFrame em arrays de inteiros
    usando os dicionários gerados por fit_cat_maps, fallback 0=UNK.
    Parâmetros:
        * df: pd.DataFrame -> dataframe.
        * cat_cols: List[str] -> lista de variáveis categóricas do dataframe.
        * maps: Dict[str,Dict[str,int]] -> dicionário {coluna: {categoria: id}} vindo do treino.
    Retorna:
        * arrs: List[np.ndarray] -> lista dos arrays resultantes. (n_amostras,)
    """
    arrs: List[np.ndarray] = []
    for c in cat_cols:
        m: Dict[str, int] = maps[c]
        s: pd.Series = df[c].astype("object").fillna("__NA__").astype(str)
        arrs.append(s.map(lambda v: m.get(v,0)).astype("int64").to_numpy())
    return arrs

def fit_text_pipe(train_text: pd.Series, hash_n: int, svd_d: int) -> Tuple[HashingVectorizer, TfidfTransformer, TruncatedSVD]:
    """
    Constrói e treina um pipeline em CPU que transforma texto cru em vetor denso de
    dimensão fixa. Utiliza:
        * HashingVectorizer (contagem, com n-gramas) (n_features=hash_n, ngramas 1–2)
        * TfidfTransformer (pesa termos raros vs frequentes)
        * TruncatedSVD (reduz dimensionalidade para svd_d)
    Parâmetros:
        * train_text: pd.Series -> série de strings (coluna de texto no treino).
        * hash_n: int -> número de features utilizadas pelo HashingVectorizer.
        * svd_d: int -> número de componentes que o vetor final irá possuir (TruncatedSVD).
    Retorna:
        * (hv, tfidf, svd): Tuple[HasshingVectorizer, TfidfTransformer, TruncatedSVD] -> os objetos ajustados, prontos para aplicar em valid/test.
    """
    
    hv: HashingVectorizer = HashingVectorizer(n_features=hash_n, alternate_sign=False, norm=None, ngram_range=(1,2))
    tfidf: TfidfTransformer = TfidfTransformer()
    svd: TruncatedSVD = TruncatedSVD(n_components=svd_d, random_state=SEED)

    Xc: csr_matrix = hv.transform(train_text.fillna(""))
    Xt: csr_matrix = tfidf.fit_transform(Xc)
    Xs: np.ndarray = svd.fit_transform(Xt)
    return hv, tfidf, svd

def transform_text(text: pd.Series, hv: HashingVectorizer, tfidf: TfidfTransformer, svd: TruncatedSVD) -> np.ndarray:
    """
    Aplica o pipeline de texto já ajustado em uma nova série de texto.
    Parâmetros:
        * text: pd.Series -> textos para transformar.
        * hv: HashingVectorizer -> modelo que converte texto cru em contagem.
        * tfidf: TfidfTransformer -> modelo que converte a contagem em pesos.
        * svd: TruncatedSVD -> modelo que reduz a dimensionalidade.
    Retorno:
        * Xs: np.ndarray -> o vetor denso gerado pelo pipeline. (n_amostras, 256)
    """
    Xc: csr_matrix = hv.transform(text.fillna(""))
    Xt: csr_matrix = tfidf.transform(Xc)
    Xs: csr_matrix = svd.transform(Xt)
    return Xs.astype("float32")

# ==================== FUNÇÃO PRINCIPAL ====================
def preprocess(
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
        vital_bounds: Dict[str, Tuple[int, int]],
        log1p_cands: List[str],
        hash_n: int,
        svd_d: int,
        *,
        text_col: Optional[str] = None,
        extra_cat: Optional[List[str]] = ["gender","triage_acuity"],
        optional_cat: Optional[List[str]] = ["race","arrival_transport","disposition","insurance","ethnicity"],
        drop_misc: Optional[List[str]] = None,
        imputation_method: str = "mean"
        ):
    """
    Realiza pipeline completo de preprocessamento.
        1) 
    Parâmetros:
        * train: pd.DataFrame -> dataframe de treino.
        * valid: pd.DataFrame -> dataframe de validação.
        * test: pd.DataFrame -> dataframe de teste.
        * target_col: str -> nome da variável (coluna) alvo.
        * vital_bounds: Dict[str, Tuple[int, int]] -> dicionário no formato {variavel: (lo, hi)}
        * log1p_cands: List[str] -> lista de variáveis com cauda longa (transformação logarítmica).
        * hash_n: int -> número de features do HashinVectorizer.
        * svd_d: int -> número de componentes do TruncatedSVD.
        * text_col: str -> nome da coluna de texto.
        * extra_cat: Optional[List[str]] -> outras features categóricas.
        * optional_cat: Optional[List[str]] -> features categóricas adicionais.
        * drop_misc: Optional[List[str]] -> features que não vão ser utilizadas no treinamento do modelo.
        * imputation_method: str -> método de imputação: "mean", "median" ou "most frequent" (esta última também pode ser passada como "mode").
    """
    # 0) Tipagem leve
    train: pd.DataFrame = train.copy()
    valid: pd.DataFrame = valid.copy()
    test: pd.DataFrame = test.copy()
    for df in (train, valid, test):
        df = prepare_types(df)

    # 1) Listas de features
    lists: Tuple[List[str], List[str], Optional[str], List[str]] = make_feature_lists(train,
                                                                target_col=target_col,
                                                                text_col=text_col,
                                                                extra_cat=extra_cat,
                                                                optional_cat=optional_cat,
                                                                drop_misc=drop_misc
                                                                )
    
    num_cols: List[str] = lists[0]
    cat_cols: List[str] = lists[1]
    text_col: Optional[str] = lists[2]
    drops = lists[3]

    for df in (train, valid, test):
        df.drop(columns=[c for c in drops if c in df.columns], inplace=True, errors="ignore")

    # 2) Regras clínicas / clamp / log1p
    clamp_limits: Dict[str, Tuple[float, float]] = apply_clinical_rules(train,
                                        valid,
                                        test,
                                        vital_bounds=vital_bounds,
                                        log1p_cands=log1p_cands
                                        )

    # 3) Imputação por média do treino nas numéricas
    train_num: pd.Series[float | int] = train[num_cols].copy()
    valid_num: pd.Series[float | int] = valid[num_cols].copy()
    test_num: pd.Series[float | int]  = test[num_cols].copy()

    if imputation_method.strip().lower() == "mean":
        imp: float = train_num.mean(axis=0)
    elif imputation_method.strip().lower() == "median":
        imp: float = train_num.median(axis=0)
    elif imputation_method.strip().lower() == "mode" or imputation_method.strip().replace("_", " ").lower() == "most frequent":
        imp: float = train_num.mode().iloc[0]
    else:
        raise ValueError("Parameter imputation_method has a unsupported value")

    for df in (train_num, valid_num, test_num):
        df.fillna(imp, inplace=True)

    # 4) Escalonamento
    num_scaler: Tuple[StandardScaler, np.ndarray] = fit_num_scaler(train_num)
    scaler: StandardScaler = num_scaler[0]
    Xnum_tr: np.ndarray = num_scaler[1]

    Xnum_va: np.ndarray = transform_num(scaler, valid_num)
    Xnum_te: np.ndarray = transform_num(scaler, test_num)

    # 5) Categóricas → ids
    cat_maps = fit_cat_maps(train, cat_cols=cat_cols) if len(cat_cols) else {}
    Xcat_tr = apply_cat_maps(train, cat_cols=cat_cols, maps=cat_maps) if len(cat_cols) else []
    Xcat_va = apply_cat_maps(valid, cat_cols=cat_cols, maps=cat_maps) if len(cat_cols) else []
    Xcat_te = apply_cat_maps(test,  cat_cols=cat_cols, maps=cat_maps)  if len(cat_cols) else []
    cat_cards = [ (max(a) + 1) if len(a)>0 else 1 for a in Xcat_tr ]  # +1 por UNK=0

    # 6) Texto
    if text_col and text_col in train.columns:
        hv, tfidf, svd = fit_text_pipe(train[text_col], hash_n=hash_n, svd_d=svd_d)
        Xtxt_tr = transform_text(train[text_col], hv=hv, tfidf=tfidf, svd=svd)
        Xtxt_va = transform_text(valid[text_col], hv=hv, tfidf=tfidf, svd=svd)
        Xtxt_te = transform_text(test[text_col],  hv=hv, tfidf=tfidf, svd=svd)
        txt_dim = Xtxt_tr.shape[1]
    else:
        hv=tfidf=svd=None
        Xtxt_tr = Xtxt_va = Xtxt_te = None
        txt_dim = None

    # 7) y e tipo
    y_tr = train[target_col].to_numpy()
    y_va = valid[target_col].to_numpy()
    y_te = test[target_col].to_numpy()

    # 8) Artefatos
    artifacts = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "text_col": text_col,
        "num_means": imp.to_dict() if imputation_method.strip().lower() == "mean" else None,
        "num_medians": imp.to_dict() if imputation_method.strip().lower() == "median" else None,
        "num_means": imp.to_dict() if imputation_method.strip().lower() == "mode" or imputation_method.strip().replace("_", " ").lower() == "most frequent" else None,
        "num_scaler": scaler,
        "cat_maps": cat_maps,
        "cat_cards": cat_cards,
        "text_pipe": {"hv": hv, "tfidf": tfidf, "svd": svd},
        "clamp_limits": clamp_limits
    }

    return (Xnum_tr, Xcat_tr, Xtxt_tr, y_tr,
            Xnum_va, Xcat_va, Xtxt_va, y_va,
            Xnum_te, Xcat_te, Xtxt_te, y_te,
            txt_dim, artifacts)