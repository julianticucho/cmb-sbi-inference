from src.simulation.api import generate_simulations

if __name__ == "__main__":
    generate_simulations(
        num_simulations=2500,
        prior_type="pol",
        simulator_type="d1s1_skymap",
        seed=1,
        num_workers=11,
        output_name="pol_d1s1_skymap_1k_1.pt",
    )
    print("simulations saved.")

    generate_simulations(
        num_simulations=2500,
        prior_type="pol",
        simulator_type="d1s1_skymap",
        seed=2,
        num_workers=11,
        output_name="pol_d1s1_skymap_1k_2.pt",
    )
    print("simulations saved.")

    generate_simulations(
        num_simulations=2500,
        prior_type="pol",
        simulator_type="d1s1_skymap",
        seed=3,
        num_workers=11,
        output_name="pol_d1s1_skymap_1k_3.pt",
    )
    print("simulations saved.")

    generate_simulations(
        num_simulations=2500,
        prior_type="pol",
        simulator_type="d1s1_skymap",
        seed=4,
        num_workers=11,
        output_name="pol_d1s1_skymap_1k_4.pt",
    )
    print("simulations saved.")

    

    
