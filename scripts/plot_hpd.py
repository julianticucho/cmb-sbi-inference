from typing import Optional, List
import torch
from src.core import StorageManager
from src.simulation import SimulatorFactory, PriorFactory
from src.preprocessing import PipelineFactory
from src.inference import NeuralInferenceFactory
from src.visualization import plot_hpd


def load_model_and_create_components(
    model_filename: str,
    simulator_type: str = "tt",
    pipeline_type: str = "planck_processing",
) -> tuple:
    storage = StorageManager()
    state_dict, simulation_files, prior_type, inference_type = storage.load_model(model_filename)
    
    theta, x = storage.load_multiple_simulations(simulation_files)
    model = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train(max_num_epochs=0)
    density_estimator.load_state_dict(state_dict)
    posterior = model.build_posterior(density_estimator)
    
    simulator = SimulatorFactory.get_simulator(simulator_type)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)
    simulator_func = lambda theta: pipeline.simulate_example(theta, simulator)
    
    prior = PriorFactory.get_prior(prior_type).to_sbi()
    
    return posterior, simulator_func, prior


def plot_hpd_and_save(
    posterior: torch.distributions.Distribution,
    simulator: callable,
    prior: torch.distributions.Distribution,
    param_labels: Optional[List[str]] = None,
    num_posterior_samples: int = 5000,
    num_true_samples: int = 100,
    output_hpd_name: Optional[str] = None
):
    print("Creating HPD coverage plot...")
    hpd_figure, coverage_results = plot_hpd(
        posterior=posterior,
        simulator=simulator,
        prior=prior,
        num_posterior_samples=num_posterior_samples,
        num_true_samples=num_true_samples,
        param_labels=param_labels
    )
    
    if output_hpd_name:
        storage = StorageManager()
        storage.save_diagnostic(hpd_figure, output_hpd_name)
        print(f"HPD plot saved as: {output_hpd_name}")
    
    return hpd_figure, coverage_results


if __name__ == "__main__":
    
    posterior, simulator, prior = load_model_and_create_components(
        model_filename="snpe_c_default_standard_test_50k_cov_binned.pth",
        simulator_type="tt",
        pipeline_type="planck_processing"
    )
    
    hpd_figure, coverage_results = plot_hpd_and_save(
        posterior=posterior,
        simulator=simulator,
        prior=prior,
        param_labels=[
            r'$\omega_b$', 
            r'$\omega_c$', 
            r'$100\theta_{MC}$', 
            r'$\ln(10^{10}A_s)$', 
            r'$n_s$'
        ],
        num_posterior_samples=2000,
        num_true_samples=1000,
        output_hpd_name="hpd_snpe_c_default_standard_test_50k_cov_binned.pdf"
    )