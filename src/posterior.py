import torch
from sbi.inference import SNPE_C, NPSE, MCMCPosterior, likelihood_estimator_based_potential
from sbi.utils.user_input_checks import process_prior
from src.simulator import create_simulator
from src.prior import get_prior

def posterior_SNPE_C(filename):
    prior = get_prior()
    prior, _, _ = process_prior(prior)

    density_estimator = torch.load(filename)
    inference = SNPE_C(prior=prior)
    posterior = inference.build_posterior(density_estimator)

    return posterior

def posterior_NPSE(filename, theta, x):
    prior = get_prior()
    prior, _, _ = process_prior(prior)

    inference = NPSE(prior=prior)
    inference.append_simulations(theta, x)
    density_estimator = inference.train(max_num_epochs=0)
    density_estimator.load_state_dict(torch.load(filename))
    posterior = inference.build_posterior(density_estimator)

    return posterior

def sample_posterior(posterior, true_parameter, type_str="TT+EE+BB+TE", num_samples=24000):
    simulator = create_simulator(type_str)
    Cl_obs = simulator(true_parameter)
    samples = posterior.set_default_x(Cl_obs).sample((num_samples,))

    return samples

def sampler_mcmc(likelihood_estimator, true_parameter, num_samples=1000):
    """ Perform MCMC sampling using the given likelihood estimator and prior """
    simulator = create_simulator()
    prior = get_prior()
    prior, _, _ = process_prior(prior)
    
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


