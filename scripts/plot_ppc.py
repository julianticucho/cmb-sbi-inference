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

    x_obs = simulate_observation(
        theta_true=theta_true,
        observation_type="unbinned_planck_tt",
        seed=0
    )

    samples_0 = sample_model(
        model_filename="tsnpe_maf_mlp_2448_default_standard_unbinned_planck_processing_1000_round_1.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=1,
    )

    samples_1 = sample_model(
        model_filename="tsnpe_maf_mlp_2448_default_standard_unbinned_planck_processing_2000_round_2.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=2,
    )

    samples_2 = sample_model(
        model_filename="tsnpe_maf_mlp_2448_default_standard_unbinned_planck_processing_4000_round_3.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=3,
    )

    samples_3 = sample_model(
        model_filename="tsnpe_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=4,
    )

    samples_4 = sample_model(
        model_filename="tsnpe_maf_mlp_2448_default_standard_unbinned_planck_processing_14000_round_5.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=5,
    )

    plot_and_save_ppc(
        samples=[samples_0, samples_1, samples_2, samples_3, samples_4], 
        true_parameter=theta_true.tolist(),
        param_names=['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        sample_labels=["1k sims round 1", "2k sims round 2", "4k sims round 3", "6k sims round 4", "14k sims round 5"],
        sample_colors=['#cccccc', '#999999', '#666666', '#333333', '#000000'],
        filled=[True, False, False, False, False],
        limits=[
            [0.02212-0.00022*5, 0.02212+0.00022*5],
            [0.1206-0.0021*5, 0.1206+0.0021*5],
            [1.04077-0.00047*5, 1.04077+0.00047*5],
            [3.04-0.016*5, 3.04+0.016*5],
            [0.9626-0.0057*5, 0.9626+0.0057*5]
        ],
        output_name="tsnpe_maf_mlp_2448_default_standard_unbinned_planck_processing_14000_round_(1,2,3,4,5).pdf"
    ) 

    samples_0 = sample_model(
        model_filename="tsnpe_default_standard_unbinned_planck_processing_10000_round_5.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=5,
    )

    samples_1 = sample_model(
        model_filename="tsnpe_default_standard_unbinned_planck_processing_14000_round_5.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=5,
    )

    samples_2 = sample_model(
        model_filename="tsnpe_default_standard_unbinned_planck_processing_18000_round_5.pth",
        x_obs=x_obs,
        num_samples=25000,
        round_index=5,
    )

    plot_and_save_ppc(
        samples=[samples_0, samples_1, samples_2], 
        true_parameter=theta_true.tolist(),
        param_names=['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        sample_labels=['10k sims round 5', '14k sims round 5', '18k sims round 5'],
        sample_colors=['#cccccc', '#666666', '#000000'],
        filled=[True, False, False],
        # limits=[
        #     [0.02212-0.00022*5, 0.02212+0.00022*5],
        #     [0.1206-0.0021*5, 0.1206+0.0021*5],
        #     [1.04077-0.00047*5, 1.04077+0.00047*5],
        #     [3.04-0.016*5, 3.04+0.016*5],
        #     [0.9626-0.0057*5, 0.9626+0.0057*5]
        # ],
        output_name="tsnpe_default_standard_unbinned_planck_processing_(10000,14000,18000)_round_5.pdf"
    ) 

  

