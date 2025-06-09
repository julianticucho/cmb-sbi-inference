import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

import torch
import numpy as np
import seaborn as sns
import os
from sbi.analysis import pairplot
from src.posterior import posterior_NPSE, posterior_SNPE_C, sample_posterior, sampler_mcmc
from src.config import PATHS
from src.simulator import Cl_XX

def plot_posterior(samples, true_parameter, limits, param_names, title='Posterior'):
    fig, axes = pairplot(
        samples,
        points=true_parameter,
        figsize=(10, 10),
        limits=limits,
        labels=param_names
    )
    plt.suptitle(title)
    plt.tight_layout()

    return fig, axes

def correlation_matrix(samples, param_names, title='Correlation matrix'):
    samples_np = samples.numpy()  
    correlation_matrix = np.corrcoef(samples_np.T)  

    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, xticklabels=param_names, yticklabels=param_names, vmin=-1, vmax=1)
    plt.title(title)

    return fig

if __name__ == "__main__":
    limits = torch.tensor([
        [0.02212-0.00022, 0.02212+0.00022],    
        [0.1206-0.0021, 0.1206+0.0021],  
        [1.04077-0.00047, 1.04077+0.00047],      
        [3.04-0.016, 3.04+0.016],    
        [0.9626-0.0057, 0.9626+0.0057],  
    ])
    param_names = [r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$']
    
    simulations = torch.load(os.path.join(PATHS["simulations"], "all_Cls_100000.pt"), weights_only=True)
    theta, x = simulations["theta"], Cl_XX(simulations["x"], "TT+EE")

    posterior = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT+EE_100000.pth"), theta, x)
    true_parameter = torch.tensor([0.02212, 0.1206, 1.04077, 3.04, 0.9626])
    samples = sample_posterior(posterior, true_parameter, type_str="TT+EE")

    fig, axes = plot_posterior(samples, true_parameter, limits, param_names, title='Posterior')
    plt.savefig(os.path.join(PATHS["posteriors"], "01_NPSE_TT+EE_100000.png"), dpi=300, bbox_inches='tight')

    fig = correlation_matrix(samples, param_names, title='Correlation matrix')
    plt.savefig(os.path.join(PATHS["correlation"], "NPSE_TT+EE_100000.png"), dpi=300, bbox_inches='tight')
