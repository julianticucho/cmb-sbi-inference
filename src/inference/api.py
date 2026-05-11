import torch
from typing import Optional, List, Dict
from ..simulation import PriorFactory, SimulatorFactory, PipelineFactory
from .factories import NeuralInferenceFactory, MCMCInferenceFactory
from ..core import storage
from cobaya.run import run
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator
from getdist import loadMCSamples

def train_model(
    input_files: List[str],
    prior_type: str,
    inference_type: str,
    output_name: Optional[str] = None
):
    # Load simulation data from list of files
    # Build neural inference model from prior and inference type
    # Append simulations to model and train
    theta, x = storage.load_multiple_simulations(input_files)
    model = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train()
    
    # Save model if output_name is provided
    # Return trained density estimator
    if output_name is not None:
        storage.save_model(density_estimator, input_files, prior_type, inference_type, output_name)
    return density_estimator

def train_sequential_model(
    x_obs: torch.Tensor,
    simulator_type: str,
    pipeline_type: str,
    prior_type: str,
    inference_type: str,
    num_rounds: int = 1,
    num_simulations_per_round: int = 100,
    num_workers: int = 1,
    output_name: Optional[str] = None
):
    # Load simulator, pipeline, prior and inference from factories
    simulator = SimulatorFactory.get_simulator(simulator_type)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)
    prior = PriorFactory.get_prior(prior_type).to_sbi()
    inference = NeuralInferenceFactory.get_inference(inference_type, prior_type)

    # Create simulator function for SBI
    simulator_func = lambda theta: pipeline.simulate_example(theta, simulator)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_func = process_simulator(simulator_func, prior, prior_returns_numpy)
    
    # Set initial proposal and initialize history lists
    # Create master filename suffix
    proposal = prior
    posteriors = []
    state_dicts = []
    simulation_filenames = []
    if output_name is None:
        output_name = f"{inference_type}_{prior_type}_{simulator_type}_{pipeline_type}_sequential"

    for i in range(num_rounds):
        # Simulate for SBI and save individual simulation file for this round
        theta, x = simulate_for_sbi(
            simulator_func, 
            proposal, 
            num_simulations=num_simulations_per_round, 
            num_workers=num_workers
        )
        round_sim_file = f"{output_name}_round_{i}.pt"
        storage.save_simulations(theta, x, round_sim_file)
        print(f"Saved simulations for round {i} to {round_sim_file}")
        simulation_filenames.append(round_sim_file)

        # Append simulations to model and train
        # then build and store posterior
        density_estimator = inference.append_simulations(
            theta, 
            x, 
            proposal=proposal
        ).train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        
        # Store current state dict
        # Update master file with cumulative history
        # Set proposal to current posterior
        state_dicts.append(density_estimator.state_dict())
        model_filename = f"{output_name}.pth"
        storage.save_sequential_model(
            state_dicts=state_dicts,
            simulation_files=simulation_filenames,
            prior_type=prior_type,
            inference_type=inference_type,
            simulator_type=simulator_type,
            pipeline_type=pipeline_type,
            x_obs=x_obs,
            filename=model_filename
        )
        print(f"Saved sequential model for round {i} to {model_filename}")
        proposal = posterior.set_default_x(x_obs)

    return density_estimator

def load_posterior(model_filename: str):
    # Load state dict, simulation files, prior type, and 
    # inference type from model filename
    cfg = storage.load_model(model_filename)
    state_dict = cfg["state_dict"]
    simulation_files = cfg["simulation_files"]
    prior_type = cfg["prior_type"]
    inference_type = cfg["inference_type"]
    
    # Load simulation data from list of files
    # Build neural inference model from prior and inference type
    # Append simulations to model and train for 0 epochs
    # Load state dict into density estimator
    theta, x = storage.load_multiple_simulations(simulation_files)
    model = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train(max_num_epochs=0)
    density_estimator.load_state_dict(state_dict)
    
    # Build and return posterior from density estimator
    posterior = model.build_posterior(density_estimator)
    return posterior

def load_prior(model_filename: str):
    # Load prior from model filename
    cfg = storage.load_model(model_filename)
    prior_type = cfg["prior_type"]
    return PriorFactory.get_prior(prior_type).to_sbi()

def load_all_sequential_posteriors(model_filename: str, round_index: Optional[int] = None):
    # Load the consolidated metadata
    cfg = storage.load_sequential_model(model_filename)
    prior_type = cfg["prior_type"]
    inference_type = cfg["inference_type"]
    x_obs = cfg["x_obs"]
    state_dicts = cfg["state_dicts"]
    simulation_files = cfg["simulation_files"]
    
    # Slice the history to the requested round
    if round_index is not None:
        if round_index >= len(state_dicts):
            raise ValueError(f"round_index {round_index} out of range (max {len(state_dicts)-1})")
        state_dicts = state_dicts[:round_index+1]
        simulation_files = simulation_files[:round_index+1]

    # Reconstruct the sequence
    prior = PriorFactory.get_prior(prior_type).to_sbi()
    inference = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    
    posteriors = []
    proposal = prior
    
    # Iterate through the history and reconstruct round by round
    # Load external simulations for this round and append to inference
    # Train for 0 epochs to set up the estimator
    # Load saved weights for this specific round
    # Build posterior and append to list
    # Update proposal for the next round
    for i in range(len(state_dicts)):
        theta, x = storage.load_simulations(simulation_files[i])
        inference.append_simulations(theta, x, proposal=proposal)
        density_estimator = inference.train(max_num_epochs=0)
        density_estimator.load_state_dict(state_dicts[i])
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_obs)
    
    return posteriors

def load_sequential_posterior(model_filename: str, round_index: Optional[int] = None):
    posteriors = load_all_sequential_posteriors(model_filename, round_index=round_index)
    return posteriors[-1]

def sample_model(
    model_filename: str,
    x_obs: torch.Tensor,
    num_samples: int = 1000,
    round_index: Optional[int] = None,
) -> torch.Tensor:
    # Load posterior from model filename
    # Set observation and return posterior samples
    if round_index is not None:
        posterior = load_sequential_posterior(model_filename, round_index=round_index)
    else:
        posterior = load_posterior(model_filename)
    return posterior.sample((num_samples,), x=x_obs)

def run_mcmc(
    config_name: str, 
    run_name: Optional[str] = None, 
    seed: Optional[int] = None, 
    mcmc_settings: Optional[Dict] = None
):
    # Load MCMC configuration from factory
    # Run MCMC with cobaya
    info, output_prefix = MCMCInferenceFactory.get_configuration(
        config_name,
        run_name=run_name or config_name,
        seed=seed,
        mcmc=mcmc_settings,
    )
    return run(info)

def load_chain(
    chain_prefix: str,
    param_names: Optional[List[str]] = None,
    ignore_rows: float = 0.3,
) -> torch.Tensor:
    # Load chain from run name
    # If param_names is None, return all parameters
    gds = loadMCSamples(chain_prefix, settings={"ignore_rows": ignore_rows})
    if param_names is None:
        return torch.tensor(gds.samples, dtype=torch.float32)
    
    # Create a mapping from parameter names to their indices in the chain
    # Check if all param_names are present in the chain
    name_to_index = {p.name: i for i, p in enumerate(gds.paramNames.names)}
    missing = [n for n in param_names if n not in name_to_index]
    if missing:
        raise ValueError(
            f"Some param_names were not found in chain: {missing}. "
            f"Available: {list(name_to_index.keys())}"
        )

    # Get the indices of the requested parameters
    # Return the chain with only the requested parameters
    idxs = [name_to_index[n] for n in param_names]
    samples = torch.tensor(gds.samples[:, idxs], dtype=torch.float32)
    return samples






    
        
    
    



