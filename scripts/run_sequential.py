from src.simulation.api import generate_simulations_from_proposal
import torch
from src.simulation.api import simulate_observation
from src.inference.api import train_sequential_model_per_round

if __name__ == "__main__":
    x_obs = simulate_observation(
        theta_true=torch.tensor([0.025, 5.045, 1.59, 1.935, -3.0]),
        observation_type="d1s1_skymap",
        seed=0,
    )

    # ronda 1
    generate_simulations_from_proposal(
        num_simulations=1000,
        simulator_type="d1s1_skymap",
        pipeline_type="identity",
        prior_type="pol",
        num_workers=11,
        seed=1,
        # only if round > 1
        previous_round_filename=None,
        output_name="d1s1_skymap_identity_pol_tsnpe_maf_cnn_skymap_round_1_1000_1.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "d1s1_skymap_identity_pol_tsnpe_maf_cnn_skymap_round_1_1000_1.pt",
        ],
        round=1,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        prior_type="pol",
        inference_type="snpe_c_maf_cnn_skymap",
        # only if round > 1
        previous_round_filename=None,
        # output name
        output_name="tsnpe_maf_cnn_skymap_pol_d1s1_skymap_identity_1000_round_1.pth",
    )





