from src.visualization.api import plot_and_save_diagnostics, plot_and_save_hpd_marginal

if __name__ == "__main__":
    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_100k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_0.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=0,
        device="cpu",
        output_name="snpe_c_default_standard_test_100k_cov_binned_(1000_0).pdf"
    )

    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_150k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_0.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=0,
        device="cpu",
        output_name="snpe_c_default_standard_test_150k_cov_binned_(1000_0).pdf"
    )

    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_50k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_1.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=1,
        device="cpu",
        output_name="snpe_c_default_standard_test_50k_cov_binned_(1000_1).pdf"
    )

    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_100k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_1.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=1,
        device="cpu",
        output_name="snpe_c_default_standard_test_100k_cov_binned_(1000_1).pdf"
    )

    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_150k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_1.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=1,
        device="cpu",
        output_name="snpe_c_default_standard_test_150k_cov_binned_(1000_1).pdf"
    )    

    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_200k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_1.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=1,
        device="cpu",
        output_name="snpe_c_default_standard_test_200k_cov_binned_(1000_1).pdf"
    )

    plot_and_save_diagnostics(
        model_filename="snpe_c_default_standard_test_250k_cov_binned.pth",
        simulation_files=["calibration_planck_processing_standard_tt_1000_1.pt"],
        run_hpd=True,
        run_tarp=True,
        num_posterior_samples=1000,
        num_references=10,
        seed=1,
        device="cpu",
        output_name="snpe_c_default_standard_test_250k_cov_binned_(1000_1).pdf"
    )