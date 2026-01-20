from typing import Optional
from src.simulation import PriorFactory, SimulatorFactory
from src.core import StorageManager


def generate_and_save_simulations(
    num_simulations: int = 100,
    prior_type: str = "standard",
    simulator_type: str = "tt",
    seed: Optional[int] = None,
    num_workers: int = 1,
    output_name: Optional[str] = None
):  
    prior = PriorFactory.get_prior(prior_type).to_sbi
    simulator = SimulatorFactory.get_simulator(simulator_type)
    
    print(f"Generating {num_simulations} simulations...")
    theta, x = simulator.simulate_batch(
        num_simulations=num_simulations,
        prior=prior,
        seed=seed,
        num_workers=num_workers
    )

    if output_name is None:
        output_name = f"{prior_type}_{simulator_type}_{num_simulations}_{seed}.pt"
    
    StorageManager().save_simulations(theta, x, output_name)
    print(f"Parameters shape: {theta.shape}")
    print(f"Simulations shape: {x.shape}")
    print(f"Saved as: {output_name}")
    
    return theta, x


if __name__ == "__main__":
    generate_and_save_simulations(
        num_simulations=10,
        prior_type="standard_tau",
        simulator_type="tt_tau",
        seed=1,
        num_workers=11
    )
