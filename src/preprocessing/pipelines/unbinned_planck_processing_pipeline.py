import torch
from ..contracts.base_pipeline import BasePipeline
from ..steps import ComponentSelectionStep, RangeCutStep, GaussianNoiseCovarianceStep
from ...data.planck import PlanckDataLoader


class UnbinnedPlanckProcessingPipeline(BasePipeline):
    
    def __init__(self):
        super().__init__("UnbinnedPlanckProcessingPipeline")
        self.cov_matrix, self.lmin, self.lmax, _, _, _, _, _, _ = PlanckDataLoader.load_planck_data()
        self.component_step = ComponentSelectionStep(components=["TT"])
        self.range_cut_step = RangeCutStep(l_min=30, l_max=2478)
        self.expanded_cov_matrix = PlanckDataLoader.expand_cov_from_binned(
            cov_bin=self.cov_matrix, 
            lmin=self.lmin, 
            lmax=self.lmax
        )
        self.noise_step = GaussianNoiseCovarianceStep(
            cov_matrix=self.expanded_cov_matrix
        )
    
    def run(self, x_clean_single: torch.Tensor) -> torch.Tensor:
        x_tt = self.component_step(x_clean_single)
        x_cut = self.range_cut_step(x_tt)
        x_processed = self.noise_step(x_cut)
        return x_processed
