import torch
from sbi.inference import SNPE_C, NPSE, FMPE, BNRE, MCMCPosterior, likelihood_estimator_based_potential
from sbi.neural_nets import posterior_nn, posterior_score_nn, flowmatching_nn
from src.simulator import create_simulator
from src.prior import get_prior
from src.embedding import get_embedding
from src.train import inference_SNPE_C, inference_NPSE, inference_FMPE

def load_and_build_posterior(inference, filename, theta=None, x=None):
    """Carga un estimador de densidad y lo convierte en un posterior"""
    if theta is not None and x is not None:
        inference.append_simulations(theta, x)
    
    density_estimator = inference.train(max_num_epochs=0)
    density_estimator.load_state_dict(torch.load(filename, weights_only=True))
    posterior = inference.build_posterior(density_estimator)

    return posterior

def posterior_SNPE_C(filename, embedding_net=None):
    """Construye un posterior SNPE_C a partir de un estimador de densidad cargado .pkl"""
    inference = inference_SNPE_C(model="nsf", embedding_net=embedding_net) 
    density_estimator = torch.load(filename)
    posterior = inference.build_posterior(density_estimator)

    return posterior

def posterior_NPSE(filename, theta, x, embedding_net=None):
    """Construye un posterior NPSE a partir de un estimador de densidad cargado .pth"""
    inference = inference_NPSE(sde_type="ve", embedding_net=embedding_net)
    posterior = load_and_build_posterior(inference, filename, theta, x)

    return posterior

def posterior_FMPE(filename, theta, x, embedding_net=None):
    """Construye un posterior FMPE a partir de un estimador de densidad cargado .pth"""
    inference = inference_FMPE(model="resnet", embedding_net=embedding_net)
    posterior = load_and_build_posterior(inference, filename, theta, x)

    return posterior

def sample_posterior(posterior, true_parameter, type_str="TT+EE+BB+TE", num_samples=24000):
    """Samplea un objeto posterior"""
    simulator = create_simulator(type_str)
    Cl_obs = simulator(true_parameter)
    samples = posterior.set_default_x(Cl_obs,).sample((num_samples,))

    return samples

def sampler_mcmc(likelihood_estimator, true_parameter, num_samples=1000):
    """Perform MCMC sampling using the given likelihood estimator and prior"""
    simulator = create_simulator()
    prior = get_prior()
    
    Cl_obs = simulator(true_parameter)
    potential_fn, parameter_transform = likelihood_estimator_based_potential(likelihood_estimator, prior, Cl_obs)

    posterior = MCMCPosterior(
        potential_fn,
        theta_transform=parameter_transform,
        proposal=prior,
        num_workers=4
    )
    samples = posterior.sample((num_samples,),)

    return samples


