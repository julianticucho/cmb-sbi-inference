import torch
from typing import List, Optional, Any
from src.core import StorageManager
from src.inference import NeuralInferenceFactory
from src.visualization import plot_hpd_tarp_diagnostics, plot_hpd


def load_model(model_filename: str):
    storage = StorageManager()
    state_dict, simulation_files, prior_type, inference_type = storage.load_model(model_filename)
    theta, x = storage.load_multiple_simulations(simulation_files)
    model = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train(max_num_epochs=0)
    density_estimator.load_state_dict(state_dict)
    posterior = model.build_posterior(density_estimator)
    return posterior

def load_simulations(simulations_filename: List[str]):
    storage = StorageManager()
    theta, x = storage.load_multiple_simulations(simulations_filename)
    return theta, x

def plot_and_save_hpd_tarp_diagnostics(
    model: Any,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    run_hpd: bool = True,
    run_tarp: bool = True,
    num_posterior_samples: int = 1000,
    num_references: int = 1,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_name: Optional[str] = None,
):
    fig = plot_hpd_tarp_diagnostics(
        model,
        thetas,
        xs,
        run_hpd=run_hpd,
        run_tarp=run_tarp,
        num_posterior_samples=num_posterior_samples,
        num_references=num_references,
        seed=seed,
        device=device,
    )
    if output_name is not None:
        storage = StorageManager()
        storage.save_diagnostics(fig, output_name)
        print(f"Diagnostic saved as: {output_name}")
    return fig

def plot_and_save_hdp(
    model: Any,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    num_posterior_samples: int = 5000,
    percentiles: Optional[torch.Tensor] = None,
    device: str = "cpu",
    param_labels: Optional[list] = None,
    output_name: Optional[str] = None,
):
    fig, results = plot_hpd(
        model,
        thetas,
        xs,
        num_posterior_samples=num_posterior_samples,
        percentiles=percentiles,
        device=device,
        param_labels=param_labels,
    )
    if output_name is not None:
        storage = StorageManager()
        storage.save_hpd(fig, output_name)
        print(f"Diagnostic saved as: {output_name}")
    return fig, results

if __name__ == "__main__":
    model = load_model("npse_regular_standard_test_100k_cov_binned.pth")
    thetas, xs = load_simulations(["calibration_planck_processing_standard_tt_1000_0.pt"])
    # plot_and_save_hdp(
    #     model=model,
    #     thetas=thetas,
    #     xs=xs,
    #     num_posterior_samples=2000,
    #     percentiles=None,
    #     device="cpu",
    #     param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
    #     output_name="npse_regular_standard_test_100k_cov_binned_(1000_0).pdf"
    # )
    plot_and_save_hpd_tarp_diagnostics(
        model=model,
        thetas=thetas,
        xs=xs,
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=2000,
        num_references=10,
        seed=0,
        device="cpu",
        output_name="npse_regular_standard_test_100k_cov_binned_(1000_0).pdf"
    )