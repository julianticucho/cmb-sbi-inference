from src.visualization.api import plot_and_save_hpd

if __name__ == "__main__":
    plot_and_save_hpd(
        model_filename="snpe_c_default_standard_test_50k_cov_binned.pth",
        simulator_type="tt",
        pipeline_type="planck_processing",
        param_labels=[
            r'$\omega_b$', 
            r'$\omega_c$', 
            r'$100\theta_{MC}$', 
            r'$\ln(10^{10}A_s)$', 
            r'$n_s$'
        ],
        num_posterior_samples=2000,
        num_true_samples=1000,
        output_name="hpd_snpe_c_default_standard_test_50k_cov_binned.pdf"
    )