import os
import torch
from sbi.inference import MCMCPosterior, likelihood_estimator_based_potential
from src.simulator.prior import get_prior
from src.simulator.simulator import create_simulator
from src.inference.utils import load_model
from src.inference.config import SBI_CONFIG

def sampler_mcmc(likelihood_estimator, prior, x_observed, num_samples):
    """
    Perform MCMC sampling using the given likelihood estimator and prior.

    """
    potential_fn, parameter_transform = likelihood_estimator_based_potential(
        likelihood_estimator, prior, x_observed
    )
    
    posterior = MCMCPosterior(
        potential_fn,
        theta_transform=parameter_transform,
        proposal=prior,
        num_workers=4
    )
    
    samples = posterior.sample(
        (num_samples,), 
    )
    
    torch.save(samples, SBI_CONFIG['sample_save_path'])
    print(f"Samples saved to {SBI_CONFIG['sample_save_path']}")

    return samples

if __name__ == "__main__":
    prior = get_prior()
    simulator = create_simulator()
    likelihood_estimator = load_model(os.path.join("results", "inference", "trained_model_17.pkl"))
    
    true_parameter = torch.tensor([0.02212, 0.1206, 1.04077, 3.04, 0.9626])
    x_observed = simulator(true_parameter)
    num_samples = 1000
    
    samples = sampler_mcmc(
        likelihood_estimator=likelihood_estimator,
        prior=prior,
        x_observed=x_observed,
        num_samples=num_samples
    )    
    print(f"Generated {len(samples)} samples")

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    samples_np = samples.numpy()  
    correlation_matrix = np.corrcoef(samples_np.T) 
    param_names = [r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'] 

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, xticklabels=param_names, yticklabels=param_names, vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.show()