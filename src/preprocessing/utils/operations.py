import torch
from typing import Tuple

def concatenate_batches(*batches: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    thetas = []
    xs = []
    for theta, x in batches:
        thetas.append(theta)
        xs.append(x)
    return torch.cat(thetas, dim=0), torch.cat(xs, dim=0)

def select_range(data: torch.Tensor, start: int, end: int, dim: int = 1) -> torch.Tensor:
    if dim == 0:
        return data[start:end]
    elif dim == 1:
        return data[:, start:end]
    else:
        raise ValueError(f"Unsupported dimension: {dim}")