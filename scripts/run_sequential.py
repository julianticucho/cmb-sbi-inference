from src.simulation.api import generate_simulations_from_proposal
import torch
from src.simulation.api import simulate_observation
from src.inference.api import train_sequential_model_per_round

if __name__ == "__main__":
    x_obs = simulate_observation(
        theta_true=torch.tensor([0.02212, 0.1206, 1.04077, 3.04, 0.9626]),
        observation_type="unbinned_planck_tt",
        seed=0,
    )

    # # ronda 1
    # generate_simulations_from_proposal(
    #     num_simulations=1000,
    #     simulator_type="tt",
    #     pipeline_type="unbinned_planck_processing",
    #     prior_type="standard",
    #     num_workers=11,
    #     seed=1,
    #     # only if round > 1
    #     previous_round_filename=None,
    #     output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_1_1000_1.pt",
    # )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_1_1000_1.pt",
        ],
        round=1,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        prior_type="standard",
        inference_type="snpe_c_maf_mlp_2448",
        # only if round > 1
        previous_round_filename=None,
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_1000_round_1.pth",
    )

    # ronda 2
    generate_simulations_from_proposal(
        num_simulations=1000,
        simulator_type="tt",
        pipeline_type="unbinned_planck_processing",
        prior_type="standard",
        num_workers=11,
        seed=1,
        truncated=True,
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_1000_round_1.pth",
        output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_2_1000_1.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_2_1000_1.pt",
        ],
        round=2,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        # prior_type="standard",
        # inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_1000_round_1.pth",
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_2000_round_2.pth",
    )

    # ronda 3
    generate_simulations_from_proposal(
        num_simulations=2000,
        simulator_type="tt",
        pipeline_type="unbinned_planck_processing",
        prior_type="standard",
        num_workers=11,
        seed=1,
        truncated=True,
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_2000_round_2.pth",
        output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_3_2000_1.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_3_2000_1.pt",
        ],
        round=3,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        # prior_type="standard",
        # inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_2000_round_2.pth",
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_4000_round_3.pth",
    )

    # ronda 4
    generate_simulations_from_proposal(
        num_simulations=2000,
        simulator_type="tt",
        pipeline_type="unbinned_planck_processing",
        prior_type="standard",
        num_workers=11,
        seed=1,
        truncated=True,
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_4000_round_3.pth",
        output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_4_2000_1.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_4_2000_1.pt",
        ],
        round=4,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        # prior_type="standard",
        # inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_4000_round_3.pth",
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
    )

    # ronda 5
    generate_simulations_from_proposal(
        num_simulations=4000,
        simulator_type="tt",
        pipeline_type="unbinned_planck_processing",
        prior_type="standard",
        num_workers=11,
        seed=1,
        truncated=True,
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_1.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_1.pt",
        ],
        round=5,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        # prior_type="standard",
        # inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_10000_round_5.pth",
    )

    generate_simulations_from_proposal(
        num_simulations=4000,
        simulator_type="tt",
        pipeline_type="unbinned_planck_processing",
        prior_type="standard",
        num_workers=11,
        seed=2,
        truncated=True,
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_2.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_1.pt",
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_2.pt",
        ],
        round=5,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        # prior_type="standard",
        # inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_14000_round_5.pth",
    )

    generate_simulations_from_proposal(
        num_simulations=4000,
        simulator_type="tt",
        pipeline_type="unbinned_planck_processing",
        prior_type="standard",
        num_workers=11,
        seed=3,
        truncated=True,
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        output_name="tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_3.pt",
    )

    train_sequential_model_per_round(
        simulation_files=[
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_1.pt",
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_2.pt",
            "tt_unbinned_planck_processing_tsnpe_snpe_c_maf_mlp_2448_default_round_5_4000_3.pt",
        ],
        round=5,
        x_obs=x_obs,
        truncated=True,
        # only if round 1
        # prior_type="standard",
        # inference_type="snpe_c_default",
        # only if round > 1
        previous_round_filename="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_6000_round_4.pth",
        # output name
        output_name="tsnpe_snpe_c_maf_mlp_2448_default_standard_unbinned_planck_processing_18000_round_5.pth",
    )






