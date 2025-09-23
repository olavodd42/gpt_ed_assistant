# scripts/validate_artifacts.py
import os, json, pickle, sys
import numpy as np

ART_DIR = "artifacts"
DS_DIR  = os.path.join(ART_DIR, "datasets")

NPZ_FILES = {
    "train": os.path.join(DS_DIR, "train_proc.npz"),
    "valid": os.path.join(DS_DIR, "valid_proc.npz"),
    "test":  os.path.join(DS_DIR, "test_proc.npz"),
}

META_JSON      = os.path.join(ART_DIR, "meta.json")
SCALER_PKL     = os.path.join(ART_DIR, "num_scaler.pkl")
TEXT_PIPE_PKL  = os.path.join(ART_DIR, "text_pipe.pkl")
CAT_MAPS_JSON  = os.path.join(ART_DIR, "cat_maps.json")

def ok(msg):   print(f"✅ {msg}")
def warn(msg): print(f"⚠️  {msg}")
def bad(msg):  print(f"❌ {msg}")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 0) meta.json
    if not os.path.exists(META_JSON):
        bad(f"meta.json não encontrado em {META_JSON}")
        sys.exit(1)
    meta = load_json(META_JSON)
    req_meta_keys = ["num_cols","cat_cols","text_col","cat_cards","txt_dim"]
    for k in req_meta_keys:
        if k not in meta:
            bad(f"meta.json sem a chave obrigatória: {k}")
            sys.exit(1)
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    txt_dim  = int(meta.get("txt_dim", 0))
    n_cats   = len(cat_cols)
    ok(f"meta.json OK | num_cols={len(num_cols)}, cat_cols={len(cat_cols)}, txt_dim={txt_dim}")

    # 1) .npz (train/valid/test)
    for split, path in NPZ_FILES.items():
        if not os.path.exists(path):
            bad(f"{split}_proc.npz não encontrado em {path}")
            sys.exit(1)
        data = np.load(path, allow_pickle=False)
        # chaves mínimas
        needed = {"Xnum","y","has_txt","Xtxt"}
        missing = [k for k in needed if k not in data.files]
        if missing:
            bad(f"{split}: faltam chaves {missing} em {path}")
            sys.exit(1)
        # Xcat_*
        for i in range(n_cats):
            key = f"Xcat_{i}"
            if key not in data.files:
                bad(f"{split}: chave {key} ausente no NPZ")
                sys.exit(1)
        # shapes/dtypes
        Xnum = data["Xnum"]; y = data["y"]
        has_txt = bool(data["has_txt"])
        Xtxt = data["Xtxt"] if has_txt else None
        N = Xnum.shape[0]
        # y
        if y.ndim != 1 or y.shape[0] != N:
            bad(f"{split}: y deve ser 1-D com N={N}; shape atual {y.shape}")
            sys.exit(1)
        # Xnum
        if Xnum.dtype != np.float32:
            warn(f"{split}: Xnum dtype={Xnum.dtype}, ideal=float32")
        # Xtxt
        if has_txt:
            if Xtxt is None or Xtxt.shape[0] != N:
                bad(f"{split}: Xtxt inconsistente com N={N} (has_txt=True)")
                sys.exit(1)
            if txt_dim and Xtxt.shape[1] != txt_dim:
                bad(f"{split}: Xtxt.shape[1]={Xtxt.shape[1]} difere de txt_dim={txt_dim} do meta.json")
                sys.exit(1)
            if Xtxt.dtype != np.float32:
                warn(f"{split}: Xtxt dtype={Xtxt.dtype}, ideal=float32")
        # Xcat_i
        for i in range(n_cats):
            Xi = data[f"Xcat_{i}"]
            if Xi.ndim != 1 or Xi.shape[0] != N:
                bad(f"{split}: Xcat_{i} shape inválido; esperado (N,), obtido {Xi.shape}")
                sys.exit(1)
            if not np.issubdtype(Xi.dtype, np.integer):
                warn(f"{split}: Xcat_{i} dtype={Xi.dtype}, ideal=int64")
        ok(f"{split}_proc.npz OK | N={N}, F_num={Xnum.shape[1]}, F_cat={n_cats}, has_txt={has_txt}")

    # 2) num_scaler.pkl
    if not os.path.exists(SCALER_PKL):
        bad(f"num_scaler.pkl não encontrado em {SCALER_PKL}")
        sys.exit(1)
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    # checar atributos do StandardScaler
    attrs = ["mean_","scale_"]
    for a in attrs:
        if not hasattr(scaler, a):
            bad(f"num_scaler.pkl parece não ser um StandardScaler (sem atributo {a})")
            sys.exit(1)
    if len(scaler.mean_) != len(num_cols):
        bad(f"num_scaler.mean_ len={len(scaler.mean_)} difere de num_cols={len(num_cols)}")
        sys.exit(1)
    ok("num_scaler.pkl OK (StandardScaler)")

    # 3) text_pipe.pkl
    if not os.path.exists(TEXT_PIPE_PKL):
        warn("text_pipe.pkl não encontrado — ok se você não usa texto.")
    else:
        with open(TEXT_PIPE_PKL, "rb") as f:
            tp = pickle.load(f)
        for k in ("hv","tfidf","svd"):
            if k not in tp:
                bad(f"text_pipe.pkl: chave '{k}' ausente")
                sys.exit(1)
        # smoke test: transformar um texto e checar dimensão
        try:
            Xc = tp["hv"].transform(["dummy text"])
            Xt = tp["tfidf"].transform(Xc)
            Xs = tp["svd"].transform(Xt)
            if txt_dim and Xs.shape[1] != txt_dim:
                bad(f"text_pipe.pkl: svd.transform gerou dim={Xs.shape[1]}, mas meta.txt_dim={txt_dim}")
                sys.exit(1)
            ok(f"text_pipe.pkl OK (svd_dim={Xs.shape[1]})")
        except Exception as e:
            bad(f"text_pipe.pkl falhou no transform: {e}")
            sys.exit(1)

    # 4) cat_maps.json
    if not os.path.exists(CAT_MAPS_JSON):
        bad(f"cat_maps.json não encontrado em {CAT_MAPS_JSON}")
        sys.exit(1)
    cat_maps = load_json(CAT_MAPS_JSON)
    # um mapa por coluna categórica
    for col in cat_cols:
        if col not in cat_maps:
            bad(f"cat_maps.json sem mapa para coluna categórica '{col}'")
            sys.exit(1)
        m = cat_maps[col]
        # ids devem ser inteiros (1..C); 0 fica para UNK em tempo de execução
        try:
            ids = [int(v) for v in m.values()]
        except Exception:
            bad(f"cat_maps[{col}] contém ids não inteiros")
            sys.exit(1)
        if any(i <= 0 for i in ids):
            bad(f"cat_maps[{col}] deve mapear categorias para ids >= 1 (0 é reservado p/ UNK)")
            sys.exit(1)
    ok("cat_maps.json OK")

    print("\n🎉 Tudo certo! Artifacts e datasets validados com sucesso.")

if __name__ == "__main__":
    main()
