from typing import Any, Dict, Optional
from cobaya.run import run
from src.inference.factories import MCMCInferenceFactory


def run_mcmc(
    config_name: str,
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
    mcmc: Optional[Dict[str, Any]] = None,
):
    info, output_prefix = MCMCInferenceFactory.get_configuration(
        config_name,
        run_name=run_name or config_name,
        seed=seed,
        mcmc=mcmc,
    )

    print(f"Running Cobaya config '{config_name}'")
    print(f"Output prefix: {output_prefix}")
    
    updated_info, sampler = run(info)
    return updated_info, sampler


if __name__ == "__main__":
    run_mcmc(
        config_name="planck_tt_gaussian",
        run_name="planck_tt_gaussian_run_3",
        seed=3,
        mcmc={},
    )

    run_mcmc(
        config_name="planck_tt_gaussian",
        run_name="planck_tt_gaussian_run_4",
        seed=4,
        mcmc={},
    )
