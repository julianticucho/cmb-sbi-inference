import camb
import numpy as np
import torch
from typing import List
from ..contracts.base_simulator import BaseSimulator


class PowerSpectrumSimulator(BaseSimulator):
    
    COMPONENT_SLICES = {
        "TT": slice(0, 2551),      # Temperature power spectrum
        "EE": slice(2551, 5102),    # E-mode polarization power spectrum  
        "BB": slice(5102, 7653),    # B-mode polarization power spectrum
        "TE": slice(7653, 10204),  # Temperature-E mode cross correlation
    }
    
    def __init__(self, components: List[str]):
        super().__init__(data_type='power_spectrum')
        self.components = components
        for comp in components:
            if comp not in self.COMPONENT_SLICES:
                raise ValueError(f"Unknown component: {comp}. Available: {list(self.COMPONENT_SLICES.keys())}")
    
    @staticmethod
    def _compute_full_spectrum(params: List[float]) -> np.ndarray:
        ombh2, omch2, theta_MC_100, ln_10_10_As, ns = params
        pars = camb.CAMBparams()
        pars.set_cosmology(
            ombh2=ombh2,
            omch2=omch2,
            tau=0.0522,  # Fixed optical depth
            cosmomc_theta=theta_MC_100/100
        )
        pars.InitPower.set_params(ns=ns, As=np.exp(ln_10_10_As)/1e10)
        pars.set_for_lmax(2500)  # Multipole range 2-2500
        pars.set_accuracy(AccuracyBoost=1.0)
        pars.NonLinear = camb.model.NonLinear_both
        pars.WantLensing = True
        results = camb.get_results(pars)
        spectra = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
        
        return np.concatenate([spectra[:, i] for i in range(4)])
    
    def _split_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        selected = []
        for comp in self.components:
            slice_idx = self.COMPONENT_SLICES[comp]
            selected.append(spectrum[slice_idx])
        
        return np.concatenate(selected)
    
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        if parameters.ndim != 1:
            raise ValueError("Parameters must have 1 dimension")
        full_spectrum = self._compute_full_spectrum(parameters.tolist())
        selected_spectrum = self._split_spectrum(full_spectrum)
        
        return torch.from_numpy(selected_spectrum).float()