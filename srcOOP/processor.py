import torch
import numpy as np
import os
from srcOOP.config import PARAMS, PATHS
from srcOOP.generator import Generator
from srcOOP.binner import bin_simulations

class Processor:
    """Postprocess a batch of CMB power spectra: noise, binning and K realizations"""

    def __init__(self, type_str="TT+EE+BB+TE", K=1, device="cpu"):
        """
        Parameters
        ----------
        type_str : str, optional
            type of spectra to process (TT, TE, EE, BB), by default "TT"
        K : int, optional
            number of noise realizations, by default 1
        device : str, optional
            device to use for computation (cpu or cuda), by default "cpu"
        """
        self.type_str = type_str
        self.K = K
        self.device = device

    def add_instrumental_noise(self, x, theta_fwhm=None, sigma_T=None):
        """Add instrumental noise to a batch of CMB power spectra x.shape=(N, L)"""

        theta_fwhm = theta_fwhm or PARAMS["noise"]["theta_fwhm"]
        sigma_T = sigma_T or PARAMS["noise"]["sigma_T"]
        ell = torch.arange(x.shape[1], device=x.device)
        theta_rad = theta_fwhm * np.pi / (180*60)
        Nl = (theta_rad * sigma_T)**2 * torch.exp(ell*(ell+1)*(theta_rad**2)/(8*np.log(2)))
        return x + Nl

    def sample_observed_spectra(self, x, l_transition=None, f_sky=None):
        """ Sample a batch of CMB power spectra x.shape=(N, L)"""

        noise_config = PARAMS["noise"]
        l_transition = l_transition or noise_config["l_transition"]
        f_sky = f_sky or noise_config["f_sky"]

        N, L = x.shape
        ell = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x_noisy = torch.empty_like(x)

        # --- high multipoles (ℓ >= l_transition): gaussian approx ---
        var_high = 2 * x[:, l_transition:]**2 / (f_sky * (2 * ell[:, l_transition:] + 1))
        noise_high = torch.sqrt(var_high) * torch.randn_like(var_high)
        x_noisy[:, l_transition:] = x[:, l_transition:] + noise_high

        # --- low multipoles (ℓ < l_transition): chi-square ---
        dof = (f_sky * (2 * ell[:, :l_transition] + 1)).round().clamp(min=1).long()
        max_dof = dof.max().item()
 
        samples = torch.randn((N, l_transition, max_dof), device=x.device)
        samples = samples * torch.sqrt(x[:, :l_transition].unsqueeze(-1))  # (N, l_transition, max_dof)
        samples_sq = samples**2

        noisy_low = torch.empty((N, l_transition), device=x.device)
        for ell_idx in range(l_transition):
            d = dof[0, ell_idx].item()  
            noisy_low[:, ell_idx] = samples_sq[:, ell_idx, :d].mean(dim=-1)

        x_noisy[:, :l_transition] = noisy_low
        return x_noisy

    def bin_spectra(self, x, bin_width=None):
        """ Apply binning to a batch of CMB power spectra"""

        bin_width = bin_width or PARAMS["binning"].get(self.type_str, 500)
        N, L = x.shape
        x_binned = []
        for i in range(N):
            spec = x[i].unsqueeze(0)  # shape (1,L)
            binned = bin_simulations(spec, 0, L-1, bin_width)[1].squeeze(0)
            x_binned.append(binned)
        return torch.stack(x_binned, dim=0)

    def generate_noise_multiple(self, theta, x):
        """ Generate K noise realizations for a batch of CMB power spectra, returning a tensor (N*K, L)"""

        x = x.to(self.device)
        N, L = x.shape
        x_expanded = x.repeat_interleave(self.K, dim=0)
        theta_expanded = theta.repeat_interleave(self.K, dim=0)
   
        x_noisy = self.add_instrumental_noise(x_expanded)
        x_noisy = self.sample_observed_spectra(x_noisy)
        return theta_expanded, x_noisy
    
    def create_simulator(self, noise: bool=False, binning=False):
        """Return a simulator function for SBI (Cls with noise and binning)"""
        def simulator(theta):
            if isinstance(theta, (list, np.ndarray)):
                theta = torch.tensor(theta, dtype=torch.float32)
            if theta.ndim == 1:
                theta = theta.unsqueeze(0)

            gen = Generator(type_str=self.type_str, num_workers=1, seed=1)
            full_spec = gen.compute_spectrum(theta[0].tolist())  # de momento solo 1 simulación
            comps = gen.split_spectra(full_spec)
            selected = [comps[c] for c in self.type_str.split("+") if c in comps]
            spec = torch.from_numpy(np.concatenate(selected)).float().unsqueeze(0)  # (1,L)

            if noise:
                _, spec = self.generate_noise_multiple(theta, spec)

            if binning:
                spec = self.bin_spectra(spec, bin_width=binning)

            return spec[0]

        return simulator
    
    def select_components(self, x: torch.Tensor, TT: bool=False, EE: bool=False, BB: bool=False, TE: bool=False) -> torch.Tensor:
        """ Select components of the CMB power spectrum (TT, EE, BB, TE)"""
        COMPONENT_SLICES = {
            "TT": slice(0, 2551),
            "EE": slice(2551, 5102),
            "BB": slice(5102, 7653),
            "TE": slice(7653, 10204),
        }

        selected_parts = []
        if TT:
            selected_parts.append(x[:, COMPONENT_SLICES["TT"]])
        if EE:
            selected_parts.append(x[:, COMPONENT_SLICES["EE"]])
        if BB:
            selected_parts.append(x[:, COMPONENT_SLICES["BB"]])
        if TE:
            selected_parts.append(x[:, COMPONENT_SLICES["TE"]])

        if not selected_parts:
            raise ValueError("Must to select at least one component (TT, EE, BB, TE).")

        return torch.cat(selected_parts, dim=1)
    
    def save_simulations(self, theta: torch.Tensor, x, name: str):
        """Save a batch of CMB power spectra x.shape=(N, L) and theta.shape=(N, P)"""
        base_path = os.path.join(PATHS["simulations"])
        tensor_dict = {"theta": theta, "x": x}
        torch.save(tensor_dict, os.path.join(base_path, name))

    def load_simulations(self, name: str):
        """Load a batch of CMB power spectra x.shape=(N, L) and theta.shape=(N, P)"""
        base_path = os.path.join(PATHS["simulations"])
        dict = torch.load(os.path.join(base_path, name), weights_only=True)
        theta, x = dict["theta"], dict["x"]
        return theta, x
