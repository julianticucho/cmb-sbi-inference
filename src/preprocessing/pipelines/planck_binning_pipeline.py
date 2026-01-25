import torch
from typing import Optional
from ..contracts.base_pipeline import BasePipeline
from ...data.planck import PlanckDataLoader
from ..steps import (
    ComponentSelectionStep, 
    RangeCutStep, 
    BinningPlanckStep
)


class PlanckBinningPipeline(BasePipeline):
    
    def __init__(self):
        super().__init__("PlanckBinningPipeline")
        _, self.lmin, self.lmax, _, _, _, _, _, _ = PlanckDataLoader.load_planck_data()
        self.component_step = ComponentSelectionStep(components=["TT"])
        self.range_cut_step = RangeCutStep(l_min=30, l_max=2478)
        self.binning_step = BinningPlanckStep(l_min=self.lmin, l_max=self.lmax)

    def run(self, x_clean_single: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        x_tt = self.component_step(x_clean_single)
        x_cut = self.range_cut_step(x_tt)
        x_binned = self.binning_step(x_cut)
        return x_binned