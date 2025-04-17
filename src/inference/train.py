import torch
from sbi.inference import SNPE_C  
from sbi.utils.user_input_checks import process_prior, process_simulator, check_sbi_inputs
from src.inference.config import SBI_CONFIG, get_prior
from src.inference.utils import load_data, preprocess_spectra, save_model

def train_sbi_model():
    spectra, params = load_data()
    processed_spectra = preprocess_spectra(spectra)
    
    prior = get_prior()
    
    def simulator(theta: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(0, len(processed_spectra), (theta.shape[0],))
        return processed_spectra[idx]
    
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    
    inference = SNPE_C(
        prior=prior,
        density_estimator=SBI_CONFIG["density_estimator"],
        device=SBI_CONFIG["device"]
    )
    
    density_estimator = inference.append_simulations(
        theta=params,
        x=processed_spectra
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