import numpy as np
from sbi.inference import SNPE_C, simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator, check_sbi_inputs
from src.simulator.prior import get_prior
from src.simulator.simulator import create_simulator
from src.inference.config import SBI_CONFIG
from src.inference.utils import save_model

def train():
    """Entrenamiento y validación del modelo sbi, junto con el cálculo de las summary statistics mediante una CNN"""
    prior = get_prior()
    simulator = create_simulator()
    
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)

    inference = SNPE_C(prior=prior)
    theta, x = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=SBI_CONFIG["num_simulations"], num_workers=11)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    
    save_model(density_estimator, SBI_CONFIG["model_save_path"])

    return density_estimator

if __name__ == "__main__":
    train()