from src.simulation.api import generate_simulations_from_proposal

if __name__ == "__main__":
    generate_simulations_from_proposal(
        num_simulations=1000,
        simulator_type="tt",
        pipeline_type="planck_binning_200",
        prior_type="standard",
        num_workers=11,
        seed=1,
        # only if round > 1
        previous_round_filename="tsnpe_default_standard_planck_binning_200_1000_round_1.pth",
        output_name="tt_planck_binning_200_tsnpe_default_round_2_1000_1.pt",
    )



