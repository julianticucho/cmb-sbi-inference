import numpy as np
from sbi.inference import SNPE_C, simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator, check_sbi_inputs
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding
from src.inference.config import SBI_CONFIG, EMBEDDING_CONFIG
from src.inference.prior import get_prior
from src.inference.simulator import create_simulator
from src.inference.utils import save_model

def train():
    """Entrenamiento y validación del modelo sbi, junto con el cálculo de las summary statistics mediante una CNN"""
    prior = get_prior()
    simulator = create_simulator()
    
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)

    embedding_net = CNNEmbedding(**EMBEDDING_CONFIG)
    neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net)

    inference = SNPE_C(
        prior=prior,
        density_estimator=neural_posterior,
        device=SBI_CONFIG["device"]
    )
    
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=SBI_CONFIG["num_simulations"])

    density_estimator = inference.append_simulations(
        theta=theta,
        x=x
    ).train(
        training_batch_size=SBI_CONFIG["training_batch_size"],
        max_num_epochs=SBI_CONFIG["training_epochs"],
        validation_fraction=SBI_CONFIG["validation_fraction"],
        show_train_summary=True  
    )
    
    save_model(density_estimator, SBI_CONFIG["model_save_path"])

    return density_estimator

if __name__ == "__main__":
    train()