import torch
from src.simulation.api import simulate_observation
from src.inference.api import sample_model, load_chain
from src.visualization.api import plot_and_save_ppc

if __name__ == "__main__":
    theta_true = torch.tensor([
        0.02212,    # ombh2
        0.1206,     # omch2
        1.04077,    # theta_MC_100
        3.04,       # ln_10_10_As
        0.9626      # ns
    ])

    x_obs_list = []
    samples_list = []
    filled = []
    colors = []

    for seed in range(1, 10):
        x_obs = simulate_observation(
            theta_true=theta_true,
            observation_type="unbinned_planck_tt",
            seed=seed
        )
        x_obs_list.append(x_obs)

        samples = sample_model(
            model_filename="snpe_c_default_standard_test_50k_cov_unbinned.pth", 
            x_obs=x_obs,
            num_samples=25000
        )
        samples_list.append(samples)
        filled.append(False)
        colors.append("#777777")

    # mcmc_samples = load_chain(
    #     chain_prefix="results/chains/planck_tt_gaussian_run_4",
    #     param_names=["ombh2", "omch2", "theta_MC_100", "ln_10_10_As", "ns"],
    #     ignore_rows=0.3
    # )   

    plot_and_save_ppc(
        samples=samples_list,  
        true_parameter=theta_true.tolist(),
        param_names=['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        # sample_labels=['obs seed 1', 'obs seed 2', 'obs seed 3'],
        sample_colors=colors,
        filled=filled,
        # limits=[
        #     [0.02212-0.00022*1, 0.02212+0.00022*1],
        #     [0.1206-0.0021*1, 0.1206+0.0021*1],
        #     [1.04077-0.00047*1, 1.04077+0.00047*1],
        #     [3.04-0.016*1, 3.04+0.016*1],
        #     [0.9626-0.0057*1, 0.9626+0.0057*1]
        # ],
        output_name="snpe_c_default_standard_test_50k_cov_unbinned(10_obs_seeds).pdf"
    ) 
