import torch
import numpy as np
from sbi.inference import SNPE_C  
from sbi.utils.user_input_checks import process_prior, process_simulator, check_sbi_inputs
from src.inference.config import SBI_CONFIG, get_prior
from src.inference.utils import load_data, preprocess_spectra, save_model
from src.simulation.generate_cosmologies import compute_spectrum
from src.simulation.add_noise import add_instrumental_noise, sample_observed_spectra
from src.simulation.config import PARAMS as SIM_PARAMS

def create_simulator():
    def simulator(theta: torch.Tensor) -> torch.Tensor:
        batch_spectra = []
        
        for i, params in enumerate(theta):
            print(f"Simulación {i+1}/{len(theta)} - Parámetros: {params.numpy()}")
            spectrum = compute_spectrum(params.numpy())
            spectrum = spectrum[:SIM_PARAMS["cosmologies"]["lmax"]]
            noisy_spectrum = add_instrumental_noise(spectrum[np.newaxis, :])[0]
            observed_spectrum = sample_observed_spectra(noisy_spectrum[np.newaxis, :])[0]
            batch_spectra.append(observed_spectrum)
        
        batch_spectra = torch.from_numpy(np.array(batch_spectra)).float()
        return preprocess_spectra(batch_spectra)
    
    return simulator

def train_sbi_model():
    prior = get_prior()
    theta = prior.sample((SBI_CONFIG.get("num_simulations"),))
    simulator = create_simulator()
    
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    
    inference = SNPE_C(
        prior=prior,
        density_estimator=SBI_CONFIG["density_estimator"],
        device=SBI_CONFIG["device"]
    )
    
    density_estimator = inference.append_simulations(
        theta=theta,
        x=simulator(theta)
    ).train(
        training_batch_size=SBI_CONFIG["training_batch_size"],
        max_num_epochs=SBI_CONFIG["training_epochs"],
        validation_fraction=SBI_CONFIG["validation_fraction"],
        show_train_summary=True  
    )
    
    save_model(density_estimator, SBI_CONFIG["model_save_path"])
    return density_estimator

if __name__ == "__main__":
    train_sbi_model()