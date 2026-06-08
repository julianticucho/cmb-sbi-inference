from typing import List
import torch
import torch.nn as nn
from ..contracts.base_model import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 5,
        hidden_dims: List[int] = [64],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers: List[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)
