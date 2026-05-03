import torch
from typing import Optional, List
from .factories import PriorFactory, SimulatorFactory, PipelineFactory, ObservationFactory
from ..core import storage

def generate_simulations(
    num_simulations: int,
    prior_type: str,
    simulator_type: str,
    seed: Optional[int] = None,
    num_workers: int = 1,
    output_name: Optional[str] = None
):  
    # Get prior and simulator from factories
    # Generate batch of simulations using simulator
    prior = PriorFactory.get_prior(prior_type).to_sbi()
    simulator = SimulatorFactory.get_simulator(simulator_type)
    theta, x = simulator.simulate_batch(
        num_simulations=num_simulations,
        prior=prior,
        seed=seed,
        num_workers=num_workers
    )

    # If no output name is provided, create one
    # Save simulations to file and return them
    if output_name is None:
        output_name = f"{prior_type}_{simulator_type}_{num_simulations}_{seed}.pt"
    storage.save_simulations(theta, x, output_name)
    return theta, x

def run_pipeline(
    input_file: List[str],
    pipeline_type: str,
    output_name: Optional[str] = None
):
    # Load clean simulations from list of files
    # Get pipeline from factory and run it on simulations
    theta, x_clean = storage.load_multiple_simulations(input_file)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)   
    x_processed = pipeline.run_batch(x_clean)
    
    # If output name is provided, save processed simulations to file
    # Return processed simulations
    if output_name:
        storage.save_simulations(theta, x_processed, output_name)
    return theta, x_processed

def simulate_observation(
    theta_true: torch.Tensor,
    observation_type: str,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # Get observation from factory
    # Simulate observation and return it
    observation = ObservationFactory.get_observation(observation_type)
    return observation(theta_true, seed=seed)
