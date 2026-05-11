import torch
from ..contracts.base_step import BaseStep


class BinningPlanckStep(BaseStep):
    
    def __init__(self, l_min: torch.Tensor, l_max: torch.Tensor):
        super().__init__("binning_planck")
        self.l_min = l_min
        self.l_max = l_max
        self.n_bins = len(l_min)
        
        if len(l_min) != len(l_max):
            raise ValueError("l_min and l_max must have the same length")
        if not torch.all(l_min < l_max):
            raise ValueError("All l_min values must be less than corresponding l_max values")

    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        binned = []
        ell_start = self.l_min[0] 
        
        for lmin_val, lmax_val in zip(self.l_min, self.l_max):
            start_idx = lmin_val - ell_start
            end_idx = lmax_val - ell_start
            
            start_idx = max(0, start_idx)
            end_idx = min(x.shape[0] - 1, end_idx)

            if start_idx < end_idx:
                slice_spectrum = x[start_idx:end_idx + 1]
                bin_value = torch.mean(slice_spectrum)
            else:
                bin_value = torch.tensor(0.0, device=x.device)
            
            binned.append(bin_value)
        
        return torch.stack(binned)        

