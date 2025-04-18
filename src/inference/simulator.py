import torch
import numpy as np
from src.inference.utils import preprocess_spectra
from src.simulation.config import PARAMS as SIM_PARAMS
from src.simulation.generate_cosmologies import compute_spectrum
from src.simulation.add_noise import add_instrumental_noise, sample_observed_spectra

def create_simulator():
    """Crea el simulador siguiendo la pipeline de src/simulation y preprocesando los datos"""
    def simulator(theta: torch.Tensor) -> torch.Tensor:
        batch_spectra = []
        
        for i, params in enumerate(theta):
            print(f"Simulación {i+1}/{len(theta)} - Parámetros: {params.numpy()}")
            spectrum = compute_spectrum(params.numpy())
            spectrum = spectrum[:SIM_PARAMS["cosmologies"]["lmax"]]
            noisy_spectrum = add_instrumental_noise(spectrum[np.newaxis, :])[0]
            observed_spectrum = sample_observed_spectra(noisy_spectrum[np.newaxis, :])[0]
            batch_spectra.append(observed_spectrum)
        
        return preprocess_spectra(np.array(batch_spectra))
    
    return simulator