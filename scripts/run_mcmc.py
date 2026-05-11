from src.inference.api import run_mcmc

if __name__ == "__main__":
    run_mcmc(
        config_name="unbinned_planck_tt_gaussian",
        run_name="unbinned_planck_tt_gaussian_run_0",
        seed=0
    )


