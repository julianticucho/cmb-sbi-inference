import torch
from typing import Optional
from ..contracts.base_pipeline import BasePipeline
from ...data import PlanckDataLoader
from ..steps import (
    ComponentSelectionStep,
    RangeCutStep,
    MultipoleBinningStep,
)

class PlanckJustBinning200Pipeline(BasePipeline):
    def __init__(self):
        super().__init__("PlanckBinning200Pipeline")
        _, lmin, lmax, _, _, _, _, _, _ = PlanckDataLoader.load_planck_data()
        self.component_step = ComponentSelectionStep(components=["TT"])
        self.range_cut_step = RangeCutStep(l_min=int(lmin[0].item()), l_max=int(lmax[-1].item()))
        self.binning_step = MultipoleBinningStep(
            l_min=int(lmin[0].item()), l_max=int(lmax[-1].item()), n_bins=200
        )

    def run(self, x_clean_single: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        x_tt = self.component_step(x_clean_single)
        x_cut = self.range_cut_step(x_tt)
        x_binned = self.binning_step(x_cut)
        return x_binned
