import torch
from typing import Optional
from ..contracts.base_pipeline import BasePipeline
from ...data.planck import PlanckDataLoader
from ..steps import (
    ComponentSelectionStep, 
    RangeCutStep, 
    BinningPlanckStep, 
    GaussianNoiseCovarianceStep
)


class PlanckProcessingPipeline(BasePipeline):
    
    def __init__(self):
        super().__init__("PlanckProcessingPipeline")
        self.cov_matrix, self.lmin, self.lmax, _, _, _, _, _, _ = PlanckDataLoader.load_planck_data()
        self.component_step = ComponentSelectionStep(components=["TT"])
        self.range_cut_step = RangeCutStep(l_min=30, l_max=2478)
        self.binning_step = BinningPlanckStep(l_min=self.lmin, l_max=self.lmax)
        self.noise_step = GaussianNoiseCovarianceStep(cov_matrix=self.cov_matrix)

    def run(self, x_clean_single: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        x_tt = self.component_step(x_clean_single)
        x_cut = self.range_cut_step(x_tt)
        x_binned = self.binning_step(x_cut)
        x_processed = self.noise_step(x_binned, seed=seed)
        return x_processed
    

