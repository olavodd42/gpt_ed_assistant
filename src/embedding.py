import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def embed_split(
                col: str,
                csv_path: str,
                out_path: str,
                model: SentenceTransformer
                ) -> None:
    """
    Gera um vetor 384-d por linha da coluna de texto e salva em .npy por split.
    Parâmetros:
        - col: str -> nome da coluna textual,
        - csv_path: str -> caminho onde o csv está salvo,
        - out_path: str -> caminho onde o .npy será salvo,
        - model: SentenceTransformer -> modelo utilizado para gerar os embeddings.
    """
    df = pd.read_csv(csv_path)
    texts = df[col].fillna('').astype(str).tolist()
    model = model.to("cuda")
    emb = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    np.save(out_path, emb)
    print(csv_path, emb.shape)