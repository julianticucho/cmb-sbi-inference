import torch
from typing import Optional, List, Tuple
from .factories import PriorFactory, SimulatorFactory, PipelineFactory, ObservationFactory
from ..core import storage
from ..inference.factories import NeuralInferenceFactory
from sbi.inference import simulate_for_sbi
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import process_prior, process_simulator

def generate_simulations(
    num_simulations: int,
    prior_type: str,
    simulator_type: str,
    seed: Optional[int] = None,
    num_workers: int = 1,
    output_name: Optional[str] = None,
    save_npy: bool = False,
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
    if save_npy:
        storage.save_simulations_numpy(theta, x, output_name)
        return theta, x
    storage.save_simulations(theta, x, output_name)
    return theta, x

def generate_simulations_from_proposal(
    num_simulations: int,
    simulator_type: str,
    pipeline_type: str,
    prior_type: str,
    num_workers: int = 1,
    output_name: Optional[str] = None,
    seed: Optional[int] = None,
    previous_round_filename: Optional[str] = None,
    truncated: bool = False,
    density_quantile: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if previous_round_filename is None:
        # First round: use the prior
        prior = PriorFactory.get_prior(prior_type).to_sbi()
        proposal = prior
    else:
        # Later rounds: reconstruct the posterior from the checkpoint chain
        chain = []
        fn = previous_round_filename
        while fn is not None:
            cfg = storage.load_model_per_round(fn)
            chain.append(cfg)
            fn = cfg.get("previous_filename")
        chain.reverse()  # round 1 → round N
        first = chain[0]
        inference_type = first["inference_type"]
        prior_type = first["prior_type"]
        prior = PriorFactory.get_prior(prior_type).to_sbi()
        inference = NeuralInferenceFactory.get_inference(inference_type, prior_type)
        proposal = prior
        for cfg_j in chain:
            theta_j, x_j = storage.load_multiple_simulations(cfg_j["simulation_files"])
            inference.append_simulations(theta_j, x_j, proposal=proposal)
            de_j = inference.train(max_num_epochs=0, force_first_round_loss=truncated)
            de_j.load_state_dict(cfg_j["state_dict"])
            x_obs = cfg_j.get("x_obs")
            posterior_j = inference.build_posterior(de_j)
            if x_obs is not None:
                posterior_j = posterior_j.set_default_x(x_obs)

            if truncated:
                reject_fn = get_density_thresholder(posterior_j, quantile=density_quantile)
                proposal = RestrictedPrior(prior, reject_fn, posterior=posterior_j, sample_with="sir")
            else:
                proposal = posterior_j

    # generate simulation function
    simulator = SimulatorFactory.get_simulator(simulator_type)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)
    simulator_func = lambda theta: pipeline.simulate_example(theta, simulator)
    prior_sbi, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_func = process_simulator(simulator_func, prior_sbi, prior_returns_numpy)

    # generate batch of simulations
    theta, x = simulate_for_sbi(
        simulator_func,
        proposal,
        num_simulations=num_simulations,
        num_workers=num_workers,
        seed=seed,
    )
    if output_name is not None:
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
    embedding_nn_filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # Get observation from factory
    # Simulate observation and return it
    observation = ObservationFactory.get_observation(observation_type)
    x_obs = observation(theta_true, seed=seed)
    if embedding_nn_filename is not None:
        from src.compression.factories import ModelFactory
        checkpoint = storage.load_embedding_nn(embedding_nn_filename)
        embedding_nn = ModelFactory.get_model(checkpoint["model_name"])
        embedding_nn.load_state_dict(checkpoint["state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_nn.eval()
        embedding_nn.to(device)
        with torch.no_grad():
            x_obs = x_obs.reshape(1, -1).to(device)  # [*, 1] or [*] -> [1, input_dim]
            x_obs = embedding_nn(x_obs).squeeze(0).cpu()  # [1, output_dim] -> [output_dim]
        print(x_obs.shape)
    return x_obs
