import torch
from src.simulation.api import simulate_observation
from src.inference.api import sample_model
from src.visualization.api import plot_and_save_data_ppc

if __name__ == "__main__":
    theta_true = torch.tensor([
        0.02212,    # ombh2
        0.1206,     # omch2
        1.04077,    # theta_MC_100
        3.04,       # ln_10_10_As
        0.9626      # ns
    ])
    
    x_obs = simulate_observation(
        theta_true=theta_true,
        observation_type="planck_tt",
        seed=0
    )

    samples = sample_model(
        model_filename="snpe_c_default_standard_test_250k_cov_binned.pth",
        x_obs=x_obs,
        num_samples=1000
    )

    plot_and_save_data_ppc(
        posterior_samples=samples,
        simulator_type="tt",
        pipeline_type="planck_processing",
        observation=x_obs,
        num_samples_for_plot=100,
        log_scale=False,
        confidence_intervals=[0.68, 0.95],
        title="PPC - SNPE-C",
        xlabel="Bin / Index",
        ylabel="CMB Power Spectrum",
        output_filename="snpe_c_default_standard_test_250k_cov_binned_data_ppc.pdf"
    )
