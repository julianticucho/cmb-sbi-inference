from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch


class BasePrior(ABC):
    
    @abstractmethod
    def sample(self, num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]: 
        pass
    
    @abstractmethod
    def to_sbi(self) -> Any:
        pass
    
    @abstractmethod
    def to_pyro(self) -> Any:
        pass
