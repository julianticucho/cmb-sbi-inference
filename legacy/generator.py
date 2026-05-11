import camb
import numpy as np
import torch
import os
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator
from src.config import PARAMS, PATHS
from src.prior import get_prior
from typing import Optional, Tuple, Dict, Callable

class Generator:
    """Class that generates CMB simulations for SBI inference.
    
    It calculates the full CMB power spectrum (TT+EE+BB+TE) using CAMB and
    splits it into its components. It also provides a simulator function for
    SBI and methods to generate simulations and save them to disk.
    """
    
    def __init__(
        self, 
        type_str: str = "TT+EE+BB+TE", 
        num_workers: int = 11, 
        seed: int = 1
    ):
        """Initialize the Generator instance with the specified parameters.
        
        Parameters
        ----------
        type_str : str, optional
            string with the components of the spectrum to generate, by default "TT+EE+BB+TE"
        num_workers : int, optional
            number of workers to use for simulation, by default 11
        seed : int, optional
            random seed, by default 1
        """
        self.type_str = type_str
        self.num_workers = num_workers
        self.seed = seed
    
    @staticmethod
    def compute_spectrum(params) -> np.ndarray:
        """Calculate the full CMB power spectrum (TT+EE+BB+TE)"""
        ombh2, omch2, theta_MC_100, ln_10_10_As, ns = params
        pars = camb.CAMBparams()
        pars.set_cosmology(
            ombh2=ombh2,
            omch2=omch2,
            tau=0.0522, #0.0925
            cosmomc_theta=theta_MC_100/100
        )
        pars.InitPower.set_params(ns=ns, As=np.exp(ln_10_10_As)/1e10)
        pars.set_for_lmax(2500)
        pars.set_accuracy(AccuracyBoost=1.0)
        pars.NonLinear = camb.model.NonLinear_both
        pars.WantLensing = True
        results = camb.get_results(pars)
        spectra = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
        return np.concatenate([spectra[:, i] for i in range(4)])
    
    @staticmethod
    def split_spectra(full_spectrum) -> Dict:
        """Split the full CMB power spectrum (TT+EE+BB+TE) into its components"""
        COMPONENT_SLICES = {
            "TT": slice(0, 2551),
            "EE": slice(2551, 5102),
            "BB": slice(5102, 7653),
            "TE": slice(7653, None)
        }
        return {k: full_spectrum[v] for k, v in COMPONENT_SLICES.items()}

    def create_simulator(self) -> Callable:
        """Return a simulator function for SBI"""
        def simulator(theta):
            full_spec = self.compute_spectrum(theta)
            comps = self.split_spectra(full_spec)
            selected = [comps[c] for c in self.type_str.split("+") if c in comps]
            return torch.from_numpy(np.concatenate(selected)).float()
        return simulator
    
    def generate_cosmologies(self, num_simulations) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate simulations for SBI and return the tensors theta and x"""
        prior = get_prior()
        simulator = self.create_simulator()
        prior, _, prior_returns_numpy = process_prior(prior)
        simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)

        theta, x = simulate_for_sbi(
            simulator_wrapper, 
            proposal=prior, 
            num_simulations=num_simulations, 
            num_workers=self.num_workers, 
            seed=self.seed
        )
        return theta, x
    
    @staticmethod
    def save_simulations(
        theta: torch.Tensor, 
        x: torch.Tensor, 
        name: str
    ):
        """Save the simulations to disk"""
        base_path = os.path.join(PATHS["simulations"])
        tensor_dict = {"theta": theta, "x": x}
        torch.save(tensor_dict, os.path.join(base_path, name))

if __name__ == "__main__":
    generator = Generator()
    theta, x = generator.generate_cosmologies(20)
    print(theta.shape, x.shape)
    generator.save_simulations(theta, x, "test.pt")