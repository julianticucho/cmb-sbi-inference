import torch
from typing import Tuple
from ..contracts.base_step import BaseStep


class ModesBinningStep(BaseStep):
    
    def __init__(self, l_min: int = 2, l_max: int = 2500, n_bins: int = 30):
        super().__init__("ModesBinningStep")
        self.l_min = l_min
        self.l_max = l_max
        self.n_bins = n_bins
        self._bin_edges = self._compute_bin_edges()

    def _compute_bin_edges(self) -> torch.Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ls = torch.arange(self.l_max + 1, device=device)
        num_modes = torch.zeros(self.l_max + 1, device=device)
        cumulative_num_modes = torch.zeros(self.l_max + 1, device=device)
        bin_edges = torch.zeros(self.n_bins + 1, dtype=torch.long, device=device)
        bin_edges[0] = self.l_min
        cumulative = 0

        for i in range(self.l_min, self.l_max + 1):
            num_modes[i] = 2 * i + 1
            cumulative += num_modes[i]
            cumulative_num_modes[i] = cumulative

        num_modes_total = num_modes.sum()
        num_modes_per_bin = num_modes_total / self.n_bins

        for i in range(1, self.n_bins + 1):
            target = num_modes_per_bin * i
            bin_edges[i] = self._find_nearest(cumulative_num_modes, target)
        
        return bin_edges
    
    def _find_nearest(self, array: torch.Tensor, value: float) -> int:
        idx = (torch.abs(array - value)).argmin()
        return idx.item()
    
    def _bin_power_spectrum(
        self,
        ell: torch.Tensor,
        cl: torch.Tensor,
        bin_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = cl.device
        bin_edges = bin_edges.to(device)
        ell = ell.to(device)
        sorted_bins, _ = torch.sort(bin_edges)
        bin_indices = torch.bucketize(ell, sorted_bins, right=False)
        
        count = torch.zeros(len(bin_edges) - 1, device=device)
        cl_bin_sum = torch.zeros(len(bin_edges) - 1, device=device)
        ell_med_sum = torch.zeros(len(bin_edges) - 1, device=device)
        
        for i in range(1, len(bin_edges)):
            mask = (bin_indices == i)
            count[i-1] = mask.sum()
            cl_bin_sum[i-1] = cl[mask].sum()
            ell_med_sum[i-1] = (ell[mask] * cl[mask]).sum()
        
        nonzero_mask = cl_bin_sum != 0
        ell_median = torch.zeros_like(ell_med_sum, dtype=torch.long)
        ell_median[nonzero_mask] = (ell_med_sum[nonzero_mask] / cl_bin_sum[nonzero_mask]).long()
        
        cl_binned = torch.zeros_like(cl_bin_sum)
        cl_binned[count != 0] = cl_bin_sum[count != 0] / count[count != 0]
        
        return ell_median, cl_binned, count
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        ell = torch.arange(x.shape[0], device=x.device) + 2
        if self.l_min > 2:
            ell = ell[self.l_min:]
            x = x[self.l_min:]
        
        if len(ell) > self.l_max + 1:
            ell = ell[:self.l_max + 1]
            x = x[:self.l_max + 1]
        
        ell_median, cl_binned, _ = self._bin_power_spectrum(ell, x, self._bin_edges)
        
        return cl_binned