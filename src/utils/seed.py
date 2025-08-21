import os
import random
import numpy as np

# TensorFlow
import tensorflow as tf

# PyTorch
import torch

# Hugging Face
from transformers import set_seed as hf_set_seed

def seed(seed: int = 42):
    """
    Define a seed global para Python, NumPy, TensorFlow, PyTorch e Hugging Face.
    Garante reprodutibilidade em experimentos.
    """
    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Menos performance, mas mais reprodutível

    # Hugging Face
    hf_set_seed(seed)

    print(f"[INFO] Global seed set to {seed}")
