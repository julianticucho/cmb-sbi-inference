import torch
import numpy as np
import os
from src.config import PARAMS, PATHS
from src.generator import Generator
from src.binner import bin_simulations
from typing import Optional, Tuple, Dict, Callable
from tqdm import tqdm

class Processor:
    """
    Postprocess a batch of CMB power spectra:
    add noise, binning, and generate multiple realizations.
    """

    COMPONENT_SLICES = {
        "TT": slice(0, 2551),
        "EE": slice(2551, 5102),
        "BB": slice(5102, 7653),
        "TE": slice(7653, 10204),
    }

    def __init__(
        self,
        type_str: str = "TT+EE+BB+TE", 
        device: str = "cpu",
    ):
        """
        Args
        ----
        type_str : str, optional
            Type of spectra to process (e.g. "TT", "TT+EE"), by default "TT+EE+BB+TE".
        device : str, optional
            Device for computation ("cpu" or "cuda"), by default "cpu".
        """
        self.type_str = type_str
        self.device = device

    def add_instrumental_noise(
        self, 
        x: torch.Tensor, 
        theta_fwhm: Optional[float] = None, 
        sigma_T: Optional[float] = None
    ) -> torch.Tensor:
        """Add instrumental noise to CMB spectra."""
        theta_fwhm = theta_fwhm or PARAMS["noise"]["theta_fwhm"]
        sigma_T = sigma_T or PARAMS["noise"]["sigma_T"]
        ell = torch.arange(x.shape[1], device=x.device)
        theta_rad = theta_fwhm * np.pi / (180*60)
        Nl = (theta_rad * sigma_T)**2 * torch.exp(ell*(ell+1)*(theta_rad**2)/(8*np.log(2)))
        return x + Nl

    def sample_observed_spectra(
        self, 
        x: torch.Tensor, 
        l_transition: Optional[int] = None, 
        f_sky: Optional[float] = None
    ) -> torch.Tensor:
        """Add sample variance to spectra using Gaussian approx (high-multipoles) 
        and χ² (low-multipoles)."""
        noise_config = PARAMS["noise"]
        l_transition = l_transition or noise_config["l_transition"]
        f_sky = f_sky or noise_config["f_sky"]

        N, L = x.shape
        ell = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x_noisy = torch.empty_like(x)

        # --- high multipoles (ell >= l_transition): gaussian approx ---
        var_high = 2 * x[:, l_transition:]**2 / (f_sky * (2 * ell[:, l_transition:] + 1))
        noise_high = torch.sqrt(var_high) * torch.randn_like(var_high)
        x_noisy[:, l_transition:] = x[:, l_transition:] + noise_high

        # --- low multipoles (ell < l_transition): chi-square ---
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

    def bin_spectra(
        self, 
        x: torch.Tensor, 
        bin_width: int=None
    ) -> torch.Tensor:
        """Apply binning to a batch of CMB power spectra."""
        bin_width = bin_width or PARAMS["binning"]["bin_width"]
        N, L = x.shape
        x_binned = []
        for i in range(N): # could be parallelized in future version
            spec = x[i].unsqueeze(0)  # shape (1,L)
            binned = bin_simulations(spec, 0, L-1, bin_width)[1].squeeze(0)
            x_binned.append(binned)
        return torch.stack(x_binned, dim=0)

    def generate_noise_multiple(
        self, 
        theta: torch.Tensor, 
        x: torch.Tensor, 
        K: int=1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate K noise realizations for a batch 
        of CMB power spectra, returning a tensor (N*K, L)."""
        x = x.to(self.device)
        x_expanded = x.repeat_interleave(K, dim=0)
        theta_expanded = theta.repeat_interleave(K, dim=0)
   
        x_noisy = self.add_instrumental_noise(x_expanded)
        x_noisy = self.sample_observed_spectra(x_noisy)
        return theta_expanded, x_noisy
    
    def create_simulator(
        self, 
        noise: bool = False, 
        binning: bool = False,
        ) -> Callable:
        """Return a simulator function for SBI (Cls with noise and binning)."""
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
                _, spec = self.generate_noise_multiple(theta, spec, K=1)

            if binning:
                bin_width = PARAMS["binning"]["bin_width"]
                spec = self.bin_spectra(spec, bin_width)

            return spec[0]

        return simulator
    
    @staticmethod
    def make_CMB_map(
        ell: torch.Tensor, 
        DlTT: torch.Tensor, 
        N: int = 2**10, 
        pix_size: float = 0.5
    ) -> torch.Tensor:
        """Generate a random CMB map from Dl (TT)."""
        device = DlTT.device  

        ClTT = DlTT * 2.0 * torch.pi / (ell * (ell + 1.0))
        ClTT = ClTT.clone()
        ClTT[0] = 0.0
        if ClTT.numel() > 1:
            ClTT[1] = 0.0

        inds = torch.linspace(-0.5, 0.5, N, device=device)
        X, Y = torch.meshgrid(inds, inds, indexing="ij")
        R = torch.sqrt(X**2 + Y**2)

        pix_to_rad = (pix_size / 60.0) * torch.pi / 180.0
        ell_scale_factor = 2.0 * torch.pi / pix_to_rad
        ell2d = R * ell_scale_factor

        ell_max = int(torch.max(ell2d).item()) + 1
        ClTT_expanded = torch.zeros(ell_max, device=device)
        ClTT_expanded[:ClTT.numel()] = ClTT
        CLTT2d = ClTT_expanded[ell2d.long()]

        random_array = torch.randn((N, N), device=device)
        FT_random_array = torch.fft.fft2(random_array)
        FT_2d = torch.sqrt(CLTT2d) * FT_random_array

        CMB_T = torch.fft.ifft2(torch.fft.fftshift(FT_2d))
        CMB_T = CMB_T / pix_to_rad
        CMB_T = torch.real(CMB_T)

        return CMB_T
    
    def generate_CMB_map(
        self, 
        x: torch.Tensor, 
        N: int = 2**10, 
        pix_size: float = 0.5
    ) -> torch.Tensor:
        """Generate CMB maps for a batch of CMB power spectra
        returning a tensor (batch_size, N, N)."""
        
        device = x.device
        batch_size = x.shape[0]
        L = x.shape[1]
        ell = torch.arange(L, device=device)
        cmb_maps = []
        
        for i in range(batch_size):
            DlTT = x[i]
            cmb_map = self.make_CMB_map(ell, DlTT, N, pix_size)
            cmb_maps.append(cmb_map)
        
        cmb_maps_batch = torch.stack(cmb_maps, dim=0)
    
        return cmb_maps_batch
    
    def bin_high_ell(
        self, 
        D_ell: torch.Tensor, 
        lmin: torch.Tensor, 
        lmax: torch.Tensor, 
    ) -> torch.Tensor:
        """Bin CMB power spectrums of CAMB like Planck"""
        D_binned = []

        ell_start = lmin[0]
        for lmin_val, lmax_val in zip(lmin, lmax):
            start_idx = lmin_val - ell_start
            end_idx = lmax_val - ell_start

            start_idx = max(0, start_idx)
            end_idx = min(D_ell.shape[0] - 1, end_idx)

            slice_D = D_ell[start_idx:end_idx + 1]
            D_bin = torch.mean(slice_D)
            D_binned.append(D_bin)

        return torch.stack(D_binned)
    
    def bin_high_ell_batch(
        self, 
        D_ell_batch: torch.Tensor, 
        lmin: torch.Tensor, 
        lmax: torch.Tensor, 
    ) -> torch.Tensor:
        """Bin CMB power spectrums of CAMB like Planck"""
        binned_list = []
        for D_ell in tqdm(D_ell_batch, desc="Binning spectra"):
            D_binned = self.bin_high_ell(D_ell, lmin, lmax)
            binned_list.append(D_binned)
            
        return torch.stack(binned_list)
    
    def add_cov_noise(
        self,
        x: torch.Tensor, 
        cov: torch.Tensor,
        seed: int = None
    ) -> torch.Tensor:
        """Generate a D_ell with correlated noise for a given covariance."""
        if seed is not None:
            torch.manual_seed(seed)

        assert x.ndim == 1
        assert cov.shape[0] == cov.shape[1] == x.shape[0]

        L = torch.linalg.cholesky(cov)
        z = torch.randn(x.shape[0])
        noise = L @ z

        return x + noise
    
    def add_cov_noise_batch(
        self,
        x: torch.Tensor, 
        cov: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a batch of D_ell with correlated noise for a given covariance."""
        batch_size = x.shape[0]
        batch_list = []

        for i in tqdm(range(batch_size), desc="Adding noise"):
            batch_list.append(self.add_cov_noise(x[i], cov))  # pasar fila i

        return torch.stack(batch_list)

    @classmethod
    def select_components(
        cls,
        x: torch.Tensor, 
        TT: bool=False, 
        EE: bool=False, 
        BB: bool=False, 
        TE: bool=False
    ) -> torch.Tensor:
        """Select specific CMB components (TT, EE, BB, TE)."""
        selected = []
        if TT: selected.append(x[:, cls.COMPONENT_SLICES["TT"]])
        if EE: selected.append(x[:, cls.COMPONENT_SLICES["EE"]])
        if BB: selected.append(x[:, cls.COMPONENT_SLICES["BB"]])
        if TE: selected.append(x[:, cls.COMPONENT_SLICES["TE"]])

        if not selected:
            raise ValueError("Must select at least one component (TT, EE, BB, TE).")

        return torch.cat(selected, dim=1)
    
    @staticmethod
    def concatenate_simulations(
        theta1: torch.Tensor, x1: torch.Tensor, 
        theta2: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate two batches of (theta, spectra)."""
        theta = torch.cat((theta1, theta2), dim=0)
        x = torch.cat((x1, x2), dim=0)

        if theta.shape[0] != x.shape[0]:
            raise ValueError("Dimensions of theta and x must match.")
        
        return theta, x
    
    @staticmethod
    def select_simulations(
        theta: torch.Tensor, 
        x: torch.Tensor, 
        start: int, 
        end: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select a range of simulations."""
        return theta[start:end], x[start:end]
    
    @staticmethod
    def select_ell(
        x: torch.Tensor, 
        start: int, 
        end: int
    ) -> torch.Tensor:
        """Select a multipole range."""
        return x[:, start:end]

    def save_simulations(
        self,
        theta: torch.Tensor, 
        x: torch.Tensor, 
        name: str
    ) -> None:
        """Save simulations to disk."""
        base_path = PATHS["simulations"]
        os.makedirs(base_path, exist_ok=True)
        torch.save({"theta": theta, "x": x}, os.path.join(base_path, name))

    def load_simulations(
        self,
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load simulations from disk."""
        base_path = PATHS["simulations"]
        dict = torch.load(os.path.join(base_path, name), weights_only=True)
        theta, x = dict["theta"], dict["x"]
        return theta, x
    

if __name__ == "__main__":
    processor = Processor(type_str="TT+EE+BB+TE")
    theta, x = processor.load_simulations("01_all_Cls_reduced_prior_50000.pt")
    theta, x = processor.select_simulations(theta, x, 0, 100)
    tt = processor.select_components(x, TT=True)
    print(theta.shape, tt.shape)

    maps = processor.generate_CMB_map(tt)
    print(maps.shape)
    processor.save_simulations(theta, maps, "testmaps.pt")
    
    