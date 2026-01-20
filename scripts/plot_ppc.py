from typing import Optional, List
import torch
from src.core import StorageManager
from src.inference.factories import InferenceFactory
from src.preprocessing.factories import ObservationFactory
from src.visualization import plot_ppc


def simulate_observation(
    theta_true: torch.Tensor,
    observation_type: str = "planck_tt",
    seed: Optional[int] = None,
) -> torch.Tensor:
    return ObservationFactory.get_observation(observation_type)(theta_true, seed=seed)


def sample_model(
    model_filename: str,
    x_obs: torch.Tensor,
    num_samples: int = 25000,
) -> torch.Tensor:
    storage = StorageManager()
    state_dict, simulation_files, prior_type, inference_type = storage.load_model(model_filename)
    theta, x = storage.load_multiple_simulations(simulation_files)
    model = InferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train(max_num_epochs=0)
    density_estimator.load_state_dict(state_dict)
    posterior = model.build_posterior(density_estimator).set_default_x(x_obs)
    samples = posterior.sample((num_samples,))
    
    return samples

def plot_ppc_and_save(
    samples: List[torch.Tensor],
    true_parameter: List[float],
    param_names: List[str],
    param_labels: Optional[List[str]] = None,
    sample_labels: Optional[List[str]] = None,
    sample_colors: Optional[List[str]] = None,
    filled: bool = True,
    title: Optional[str] = None,
    limits: Optional[List[tuple]] = None,
    output_ppc_name: Optional[str] = None
):
    print("Creating posterior predictive check plot...")
    ppc_figure = plot_ppc(
        all_samples=samples,
        true_parameter=true_parameter,
        param_names=param_names,
        param_labels=param_labels,
        sample_labels=sample_labels,
        sample_colors=sample_colors,
        filled=filled,
        title=title,
        limits=limits
    )
    
    if output_ppc_name:
        storage = StorageManager()
        storage.save_ppc(ppc_figure, output_ppc_name)
        print(f"PPC plot saved as: {output_ppc_name}")
    
    return ppc_figure


if __name__ == "__main__":
    theta_true_example = torch.tensor([0.022068, 0.12029, 1.04122, 3.098, 0.9624])
    
    x_obs = simulate_observation(
        theta_true=theta_true_example,
        observation_type="planck_tt",
        seed=0
    )
    
    samples_1 = sample_model(
        model_filename="fmpe_default_25k_cov_binned.pth",
        x_obs=x_obs,
        num_samples=25000,
    )

    plot_ppc_and_save(
        samples=[samples_1],  
        true_parameter=theta_true_example.tolist(),
        param_names=['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        sample_labels=['fmpe 25k'],
        sample_colors=['#E03424'],
        filled=[True],
        limits=[
            [0.022068-0.00022*(5), 0.022068+0.00022*(5)],    
            [0.12029-0.0021*(5), 0.12029+0.0021*(5)],  
            [1.04122-0.00047*(5), 1.04122+0.00047*(5)],      
            [3.098-0.016*(5), 3.098+0.016*(5)],    
            [0.9624-0.0057*(5), 0.9624+0.0057*(5)]
        ],
        output_ppc_name="fmpe_default_25k_cov_binned.pdf"
    )
