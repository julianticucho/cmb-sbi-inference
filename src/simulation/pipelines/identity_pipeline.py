import torch
from typing import Optional
from ..contracts.base_pipeline import BasePipeline


class IdentityPipeline(BasePipeline):
    
    def __init__(self):
        super().__init__("IdentityPipeline")
        
    def run(self, x_clean_single: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        return x_clean_single