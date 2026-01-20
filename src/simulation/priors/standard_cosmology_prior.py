import torch
from typing import Dict, Tuple, Any, Optional
from ..contracts.base_prior import BasePrior
from sbi.utils import BoxUniform


class StandardCosmologyPrior(BasePrior):
    
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]], device: str = "cpu"):
        """Initialize with parameter ranges."""
        self.parameter_ranges = parameter_ranges
        self.parameter_names = list(self.parameter_ranges.keys())
        self.sbi_prior = BoxUniform(
            low=torch.tensor([self.parameter_ranges[name][0] for name in self.parameter_names]), 
            high=torch.tensor([self.parameter_ranges[name][1] for name in self.parameter_names]),
            device=device
        )
    
    def sample(self, num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        return self.sbi_prior.sample((num_samples,), seed)
    
    def get_parameter_names(self) -> list[str]:
        return self.parameter_names.copy()
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return self.parameter_ranges.copy()

    def to_sbi(self) -> Any:
        return self.sbi_prior
