from abc import ABC, abstractmethod
from typing import Optional
from contextlib import contextmanager
import torch


class BaseStep(ABC):
    
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x: torch.Tensor, seed: Optional[int] = None, **kwargs) -> torch.Tensor:
        with self.set_seed(seed):
            return self.apply(x, **kwargs)
    
    @contextmanager
    def set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
        try:
            yield
        finally:
            pass

    @abstractmethod
    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
    

