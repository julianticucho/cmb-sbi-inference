from src.simulation.api import generate_simulations

if __name__ == "__main__":
    # generate_simulations(
    #     num_simulations=50000,
    #     prior_type="1sigma",
    #     simulator_type="tt",
    #     seed=1,
    #     num_workers=11,
    #     output_name="tt_1sigma_50000_1.pt"
    # )
    # print("simulations saved as tt_1sigma_50000_1.pt.")

    # generate_simulations(
    #     num_simulations=50000,
    #     prior_type="1sigma",
    #     simulator_type="tt",
    #     seed=2,
    #     num_workers=11,
    #     output_name="tt_1sigma_50000_2.pt"
    # )
    # print("simulations saved as tt_1sigma_50000_2.pt.")

    generate_simulations(
        num_simulations=100000,
        prior_type="standard",
        simulator_type="auxiliary_observables",
        seed=0,
        num_workers=11,
        output_name="auxiliary_observables_100000_0.pt"
    )
    print("simulations saved as auxiliary_observables_100000_0.pt")


    