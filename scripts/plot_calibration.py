from src.visualization.api import plot_and_save_diagnostics

if __name__ == "__main__":
    plot_and_save_diagnostics(
        model_filename="npse_regular_standard_test_100k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_0.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=2000,
        num_references=10,
        seed=0,
        device="cpu",
        output_name="npse_regular_standard_test_100k_cov_binned_(1000_0).pdf"
    )