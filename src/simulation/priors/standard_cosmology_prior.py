import torch
import pyro.distributions as dist
from typing import Dict, Tuple, Any, Optional
from ..contracts.base_prior import BasePrior
from sbi.utils import BoxUniform


class StandardCosmologyPrior(BasePrior):
    
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]], device: str = "cpu"):
        self.parameter_ranges = parameter_ranges
        self.device = device
        self.parameter_names = list(self.parameter_ranges.keys())
        self.low = torch.tensor([self.parameter_ranges[name][0] for name in self.parameter_names])
        self.high = torch.tensor([self.parameter_ranges[name][1] for name in self.parameter_names])
        self.sbi_prior = BoxUniform(low=self.low, high=self.high, device=self.device)
    
    def sample(self, num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        return self.sbi_prior.sample((num_samples,), seed)
    
    def get_parameter_names(self) -> list[str]:
        return self.parameter_names.copy()
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return self.parameter_ranges.copy()

    def to_sbi(self) -> Any:
        return self.sbi_prior
    
    def to_pyro(self) -> Any:
        return dist.Uniform(low=self.low, high=self.high)
