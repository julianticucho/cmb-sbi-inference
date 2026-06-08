import torch
from typing import Optional
from ..contracts.base_pipeline import BasePipeline
from ...data import PlanckDataLoader
from ..steps import (
    ComponentSelectionStep,
    RangeCutStep,
    MultipoleBinningStep,
    GaussianNoiseCovarianceStep,
)

class PlanckBinning200Pipeline(BasePipeline):
    """Pipeline that produces a 200‑dim noisy TT spectrum.
    It uses the Planck multipole range (ℓ=32‑2479), bins uniformly into 200 bins,
    and adds Gaussian noise with the 200‑bin covariance.
    """

    def __init__(self):
        super().__init__("PlanckBinning200Pipeline")
        cov_bin, lmin, lmax, _, _, _, _, _, _ = PlanckDataLoader.load_planck_data()
        self.cov_200 = PlanckDataLoader.expand_cov_to_200_bins(
            cov_bin, lmin, lmax, target_bins=200, jitter=1e-2
        )
        self.component_step = ComponentSelectionStep(components=["TT"])
        self.range_cut_step = RangeCutStep(l_min=int(lmin[0].item()), l_max=int(lmax[-1].item()))
        self.binning_step = MultipoleBinningStep(
            l_min=int(lmin[0].item()), l_max=int(lmax[-1].item()), n_bins=200
        )
        self.noise_step = GaussianNoiseCovarianceStep(self.cov_200)

    def run(self, x_clean_single: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        x_tt = self.component_step(x_clean_single)
        x_cut = self.range_cut_step(x_tt)
        x_binned = self.binning_step(x_cut)
        x_noisy = self.noise_step.apply(x_binned)
        return x_noisy
