import torch
from typing import Optional
from ..contracts.base_step import BaseStep


class RangeCutStep(BaseStep):
    
    def __init__(self, l_min: Optional[int] = None, l_max: Optional[int] = None):
        super().__init__("RangeCutStep")
        self.l_min = l_min
        self.l_max = l_max
        self._validate_ranges()

    def _validate_ranges(self):
        if self.l_min is not None and self.l_min < 0:
            raise ValueError("l_min must be non-negative")
        if self.l_max is not None and self.l_max < 0:
            raise ValueError("l_max must be non-negative")
        if self.l_min is not None and self.l_max is not None and self.l_min > self.l_max:
            raise ValueError("l_min must be less than or equal to l_max")
    
    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError("Input must be 1D tensor")
        
        start_idx = 0
        if self.l_min is not None:
            start_idx = max(0, self.l_min)
        
        end_idx = x.shape[0]
        if self.l_max is not None:
            end_idx = min(x.shape[0], self.l_max + 1)
        
        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx ({start_idx}) >= end_idx ({end_idx})")
        
        return x[start_idx:end_idx]
