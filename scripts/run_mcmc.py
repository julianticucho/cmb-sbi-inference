from src.inference.api import run_mcmc

if __name__ == "__main__":
    run_mcmc(
        config_name="binned_planck_tt_200_gaussian",
        run_name="binned_planck_tt_200_gaussian_run_0",
        seed=0
    )


