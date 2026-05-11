import torch
from src.simulation.api import simulate_observation
from src.inference.api import train_sequential_model

if __name__ == "__main__":
    x_obs = simulate_observation(
        theta_true=torch.tensor([
            0.02212,    # ombh2
            0.1206,     # omch2
            1.04077,    # theta_MC_100
            3.04,       # ln_10_10_As
            0.9626      # ns
        ]),
        observation_type="planck_tt",
        seed=0,
    )

    train_sequential_model(
        x_obs=x_obs,
        simulator_type="tt",
        pipeline_type="planck_processing",
        prior_type="standard",
        inference_type="snpe_c_default",
        num_rounds=5,
        num_simulations_per_round=10000,
        num_workers=11,
        output_name="snpe_c_default_standard_tt_planck_processing_sequential_5_10000"
    )