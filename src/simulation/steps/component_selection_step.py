import torch
from typing import List
from ..contracts.base_step import BaseStep


class ComponentSelectionStep(BaseStep):
    
    COMPONENT_SLICES = {
        "TT": slice(0, 2551),
        "EE": slice(2551, 5102),
        "BB": slice(5102, 7653), 
        "TE": slice(7653, 10204),
    }
    
    def __init__(self, components: List[str]):
        super().__init__("ComponentSelectionStep")
        self.components = components
        self._validate_components()

    def _validate_components(self):
        for comp in self.components:
            if comp not in self.COMPONENT_SLICES:
                raise ValueError(f"Unknown component: {comp}. Available: {list(self.COMPONENT_SLICES.keys())}")

    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        selected = []
        for comp in self.components:
            slice_idx = self.COMPONENT_SLICES[comp]
            selected.append(x[slice_idx])
        
        if not selected:
            raise ValueError("Must select at least one component")
        
        return torch.cat(selected, dim=0)