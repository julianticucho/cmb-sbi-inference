import torch
from src.simulation.api import simulate_observation
from src.inference.api import train_sequential_model_per_round

if __name__ == "__main__":
    x_obs = simulate_observation(
        theta_true=torch.tensor([0.02212, 0.1206, 1.04077, 3.04, 0.9626]),
        observation_type="planck_tt_200",
        seed=0,
    )

    posterior = train_sequential_model_per_round(
        simulation_files=[
            "tt_planck_binning_200_tsnpe_default_round_1_1000_1.pt",
        ],
        round=1,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        prior_type="standard",
        inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename=None,
        # output name
        output_name="tsnpe_default_standard_planck_binning_200_1000_round_1.pth",
    )


