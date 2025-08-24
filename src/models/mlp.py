from typing import Any, List
import torch
import torch.nn as nn
from src.utils.seed import seed

seed()
class MLP:
    def __init__(self, hidden_size: int, n: int):
        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n)
            )
    
    def __call__(self, inputs: List[int]):
        return self.mlp(inputs)
    