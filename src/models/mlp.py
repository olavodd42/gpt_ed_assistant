import torch
import torch.nn as nn
from src.utils.seed import seed

seed()
def mlp(hidden_size: int, n: int) -> nn.Sequential:
    return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n)
        )