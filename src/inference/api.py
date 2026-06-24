import torch
from typing import Optional, List, Dict
from ..simulation import PriorFactory, SimulatorFactory, PipelineFactory
from .factories import NeuralInferenceFactory, MCMCInferenceFactory
from ..core import storage
from cobaya.run import run
from sbi.inference import simulate_for_sbi
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import process_prior, process_simulator
from getdist import loadMCSamples

def train_model(
    input_files: List[str],
    prior_type: str,
    inference_type: str,
    embedding_nn_filename: Optional[str] = None,
    output_name: Optional[str] = None
):
    # Load simulation data from list of files
    # Build neural inference model from prior and inference type
    # Append simulations to model and train
    theta, x = storage.load_multiple_simulations(input_files)
    if embedding_nn_filename is not None:
        from ..compression.factories.model_factory import ModelFactory
        checkpoint = storage.load_embedding_nn(embedding_nn_filename)
        embedding_nn = ModelFactory.get_model(checkpoint["model_name"])
        embedding_nn.load_state_dict(checkpoint["state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_nn.eval()
        embedding_nn.to(device)
        with torch.no_grad():
            x = x.to(device)
            x = embedding_nn(x).cpu()

    print(theta.shape, x.shape)
    model = NeuralInferenceFactory.get_inference(inference_type, prior_type)
    model.append_simulations(theta, x)
    density_estimator = model.train()
    
    # Save model if output_name is provided
    # Return trained density estimator
    if output_name is not None:
        storage.save_model(
            density_estimator,
            input_files,
            prior_type,
            inference_type,
            output_name,
            embedding_nn_filename=embedding_nn_filename,
        )
    return density_estimator

def train_sequential_model_per_round(
    simulation_files: List[str],
    output_name: str,
    round: int,
    x_obs: torch.Tensor,
    prior_type: Optional[str] = None,
    inference_type: Optional[str] = None,
    previous_round_filename: Optional[str] = None,
    truncated: bool = False,
    density_quantile: float = 1e-4,
):
    theta, x = storage.load_multiple_simulations(simulation_files)
    print("llego a cargar simulaciones", theta.shape, x.shape)
    if round == 1:
        # --- First round ---
        assert prior_type is not None, "prior_type required for round 1"
        assert inference_type is not None, "inference_type required for round 1"
        prior = PriorFactory.get_prior(prior_type).to_sbi()
        inference = NeuralInferenceFactory.get_inference(inference_type, prior_type)
        proposal = prior
        first_prior_type = prior_type
        first_inference_type = inference_type
    else:
        # --- Subsequent rounds: walk the checkpoint chain ---
        assert previous_round_filename is not None, "previous_round_filename required for round > 1"
        chain = []
        fn = previous_round_filename
        while fn is not None:
            cfg = storage.load_model_per_round(fn)
            chain.append(cfg)
            fn = cfg.get("previous_filename")
        chain.reverse()  # round 1 → round N-1

        first = chain[0]
        first_prior_type = first["prior_type"]
        first_inference_type = first["inference_type"]
        prior = PriorFactory.get_prior(first_prior_type).to_sbi()
        inference = NeuralInferenceFactory.get_inference(
            first_inference_type, first_prior_type
        )

        proposal = prior
        for cfg_j in chain:
            theta_j, x_j = storage.load_multiple_simulations(cfg_j["simulation_files"])
            inference.append_simulations(theta_j, x_j, proposal=proposal)
            de_j = inference.train(max_num_epochs=0, force_first_round_loss=truncated)
            de_j.load_state_dict(cfg_j["state_dict"])
            posterior_j = inference.build_posterior(de_j).set_default_x(x_obs)

            if truncated:
                reject_fn = get_density_thresholder(posterior_j, quantile=density_quantile, num_samples_to_estimate_support=100_000)
                proposal = RestrictedPrior(
                    prior, reject_fn, posterior=posterior_j, sample_with="sir"
                )
            else:
                proposal = posterior_j

    # --- Train current round ---
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal
    ).train(force_first_round_loss=truncated)
    posterior = inference.build_posterior(density_estimator).set_default_x(x_obs)

    # --- Save checkpoint for this round ---
    if output_name:
        storage.save_model_per_round(
            model=density_estimator,
            round=round,
            simulation_files=simulation_files,
            prior_type=first_prior_type,
            inference_type=first_inference_type,
            x_obs=x_obs,
            previous_filename=previous_round_filename,
            filename=output_name,
        )
    return posterior

def load_posterior(model_filename: str):
    # Load state dict, simulation files, prior type, and 
    # inference type from model filename
    cfg = storage.load_model(model_filename)
    state_dict = cfg["state_dict"]
    simulation_files = cfg["simulation_files"]
    prior_type = cfg["prior_type"]
    inference_type = cfg["inference_type"]
    embedding_nn_filename = cfg.get("embedding_nn_filename", None)

    # Load simulation data from list of files
    theta, x = storage.load_multiple_simulations(simulation_files)

    # Apply compression if the model was trained with an embedding network
    if embedding_nn_filename is not None:
        from ..compression.factories.model_factory import ModelFactory
        checkpoint = storage.load_embedding_nn(embedding_nn_filename)
        embedding_nn = ModelFactory.get_model(checkpoint["model_name"])
        embedding_nn.load_state_dict(checkpoint["state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_nn.eval()
        embedding_nn.to(device)
        with torch.no_grad():
            x = embedding_nn(x.to(device)).cpu()

    # Build neural inference model from prior and inference type
    # Append simulations to model and train for 0 epochs to initialise architecture
    # Load saved weights into density estimator
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

def load_seq_posterior(
    model_filename: str,
    round_index: int,
    truncated: bool = False,
    density_quantile: float = 1e-4,
):
    chain = []
    fn = model_filename
    while fn is not None:
        cfg = storage.load_model_per_round(fn)
        chain.append(cfg)
        fn = cfg.get("previous_filename")
    chain.reverse()

    if round_index < 1 or round_index > len(chain):
        raise ValueError(
            f"round_index {round_index} out of range. "
            f"Chain has {len(chain)} round(s) (1‑indexed)."
        )

    relevant = chain[:round_index]
    first = relevant[0]
    prior = PriorFactory.get_prior(first["prior_type"]).to_sbi()
    inference = NeuralInferenceFactory.get_inference(
        first["inference_type"], first["prior_type"]
    )

    proposal = prior
    for cfg_j in relevant:
        theta_j, x_j = storage.load_multiple_simulations(cfg_j["simulation_files"])
        inference.append_simulations(theta_j, x_j, proposal=proposal)
        del theta_j, x_j
        de_j = inference.train(max_num_epochs=0, force_first_round_loss=truncated)
        de_j.load_state_dict(cfg_j["state_dict"])
        x_obs = cfg_j.get("x_obs")
        posterior_j = inference.build_posterior(de_j, sample_with="mcmc")
        if x_obs is not None:
            posterior_j = posterior_j.set_default_x(x_obs)

        if truncated:
            reject_fn = get_density_thresholder(posterior_j, quantile=density_quantile, num_samples_to_estimate_support=1000)
            proposal = RestrictedPrior(prior, reject_fn, posterior=posterior_j, sample_with="sir")
        else:
            proposal = posterior_j

    return proposal

def sample_model(
    model_filename: str,
    x_obs: torch.Tensor,
    num_samples: int = 1000,
    round_index: Optional[int] = None,
) -> torch.Tensor:
    # Load posterior from model filename
    # Set observation and return posterior samples
    if round_index is not None:
        posterior = load_seq_posterior(
            model_filename, 
            round_index=round_index,
            truncated=False
        )
        return posterior.sample((num_samples,))
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






    
        
    
    



