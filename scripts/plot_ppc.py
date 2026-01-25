import torch
from typing import Optional, List
from getdist import loadMCSamples

from src.core import StorageManager
from src.inference.factories import NeuralInferenceFactory
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
    model = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train(max_num_epochs=0)
    density_estimator.load_state_dict(state_dict)
    posterior = model.build_posterior(density_estimator)
    samples = posterior.sample((num_samples,), x=x_obs)
    
    return samples


def load_cobaya_chain_as_tensor(
    chain_prefix: str,
    param_names: Optional[List[str]] = None,
    ignore_rows: float = 0.3,
) -> torch.Tensor:
    gds = loadMCSamples(chain_prefix, settings={"ignore_rows": ignore_rows})
    if param_names is None:
        return torch.tensor(gds.samples, dtype=torch.float32)
    name_to_index = {p.name: i for i, p in enumerate(gds.paramNames.names)}
    missing = [n for n in param_names if n not in name_to_index]
    if missing:
        raise ValueError(
            f"Some param_names were not found in chain: {missing}. "
            f"Available: {list(name_to_index.keys())}"
        )
    idxs = [name_to_index[n] for n in param_names]
    return torch.tensor(gds.samples[:, idxs], dtype=torch.float32)


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
    theta_true_example = torch.tensor([0.02212, 0.1206, 1.04077, 3.04, 0.9626])

    x_obs = simulate_observation(
        theta_true=theta_true_example,
        observation_type="planck_tt",
        seed=0
    )

    mcmc_samples = load_cobaya_chain_as_tensor(
        chain_prefix="results/chains/planck_tt_gaussian_run_4",
        param_names=["ombh2", "omch2", "theta_MC_100", "ln_10_10_As", "ns"],
        ignore_rows=0.3
    )   

    samples_1 = sample_model(
        model_filename="snpe_c_default_standard_test_250k_cov_binned.pth",
        x_obs=x_obs,
        num_samples=25000,
    )

    samples_2 = sample_model(
        model_filename="nle_default_standard_test_250k_cov_binned.pth",
        x_obs=x_obs,
        num_samples=25000,
    )
    
    plot_ppc_and_save(
        samples=[samples_1, samples_2, mcmc_samples],  
        true_parameter=theta_true_example.tolist(),
        param_names=['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        sample_labels=['snpe_c 250k', 'nle 250k', 'mcmc'],
        sample_colors=['#cccccc', '#666666', '#000000'], #'#006FED'
        filled=[True, False, False],
        limits=[
            [0.02212-0.00022*5, 0.02212+0.00022*5],
            [0.1206-0.0021*5, 0.1206+0.0021*5],
            [1.04077-0.00047*5, 1.04077+0.00047*5],
            [3.04-0.016*5, 3.04+0.016*5],
            [0.9626-0.0057*5, 0.9626+0.0057*5]
        ],
        output_ppc_name="mcmc_vs_(nle_vs_snpe_c)_default_standard_test_250k_cov_binned.pdf"
    ) 

