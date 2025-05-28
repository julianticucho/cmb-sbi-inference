import numpy as np
import torch
import os
from sbi.utils import BoxUniform
from torch import Tensor
from sbi.inference import SNPE_C, simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator, check_sbi_inputs

def synthetic_spectrum(params):
    """
    Genera un espectro sintético de 2500 puntos basado en 5 parámetros inventados.
    """
    alpha, beta, gamma, delta, epsilon = params
    x = np.linspace(0, 25, 2500)
    spectrum = (
        alpha * np.exp(-(x - 5)**2 / (2 * beta**2))  
        + gamma * np.sin(delta * x) / (x + 1)         
        + epsilon * np.log(x + 1)                     
    )
    return spectrum

def create_simulator():
    """Crea el simulador que genera espectros sintéticos"""
    def simulator(theta):
        return synthetic_spectrum(theta)
    
    return simulator

PARAM_RANGES = {
    "alpha": (10.5-0.1, 10.5+0.1),    
    "beta": (1.75-0.1, 1.75+0.1), 
    "gamma": (2.55-0.5, 2.55+0.5),    
    "delta": (1.1-0.1, 1.1+0.1),     
    "epsilon": (1-0.1, 1+0.1)    
}

def get_prior() -> BoxUniform:
    """Genera el prior en el rango de valores especificado"""
    lows = Tensor([v[0] for v in PARAM_RANGES.values()])
    highs = Tensor([v[1] for v in PARAM_RANGES.values()])
    return BoxUniform(low=lows, high=highs)

def train_NPE(simulator, prior, num_sims):
    """Entrenamiento y validación del modelo NPE"""
    
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)

    inference = SNPE_C(prior=prior)
    theta, x = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=num_sims)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(stop_after_epochs=100)

    return density_estimator, inference

def samples(inference, density_estimator, simulator):   
    """Genera muestras del posterior"""

    posterior = inference.build_posterior(density_estimator)  
    true_parameter = torch.tensor([10.5, 1.75, 2.55, 1.1, 1])
    x_observed = simulator(true_parameter)

    return posterior.set_default_x(x_observed).sample((25000,))

if __name__ == "__main__":
    simulator = create_simulator()
    prior = get_prior()
    density_estimator, inference = train_NPE(simulator, prior, num_sims=5000)
    
    from src.inference.utils import save_model
    save_model(density_estimator, os.path.join("results", "synthetic", "model_5000.pkl"))
    
    samples = samples(inference, density_estimator, simulator)
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science', 'bright'])
    plt.rcParams['figure.dpi'] = 300

    param_names = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$']
    samples_np = samples.numpy()  
    correlation_matrix = np.corrcoef(samples_np.T)  
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, xticklabels=param_names, yticklabels=param_names, vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.savefig(os.path.join("results", "synthetic", "correlation_5000.png"), dpi=300, bbox_inches='tight')

    from sbi.analysis import pairplot

    limits = torch.tensor([
        [10.5-0.01, 10.5+0.01],    
        [1.75-0.01, 1.75+0.01],  
        [2.55-0.05, 2.55+0.05],      
        [1.1-0.01, 1.1+0.01],    
        [1-0.01, 1+0.01],  
    ])
    true_parameter = torch.tensor([10.5, 1.75, 2.55, 1.1, 1])
    fig = pairplot(
        samples,
        points=true_parameter,
        figsize=(10, 10),
        limits=limits,
        labels=param_names
    )

    plt.suptitle('Posterior')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "synthetic", "posterior_5000.png"), dpi=300, bbox_inches='tight')

    from sbi.diagnostics import run_tarp
    from sbi.utils.user_input_checks import process_prior, process_simulator
    from sbi.inference import simulate_for_sbi

    num_tarp_samples = 1000
    num_posterior_samples = 5000

    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    
    inference = SNPE_C(prior=prior)
    posterior = inference.build_posterior(density_estimator)
    thetas, xs = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=num_tarp_samples, seed=0)

    ecp, alpha = run_tarp(thetas, xs, posterior, references=None, num_posterior_samples=num_posterior_samples, use_batched_sampling=False, num_workers=11)

    from sbi.analysis.plot import plot_tarp

    plot_tarp(ecp, alpha)
    plt.savefig(os.path.join("results", "synthetic", "tarp_5000.png"), dpi=300, bbox_inches='tight')