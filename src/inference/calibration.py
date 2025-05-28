import os
import torch
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.diagnostics import run_sbc, run_tarp
from sbi.inference import SNPE_C, simulate_for_sbi
from src.simulator.simulator import create_simulator
from src.simulator.prior import get_prior
from src.inference.utils import load_model
from src.inference.config import SBI_CONFIG

def sbc(prior, simulator, density_estimator, num_sbc_samples, num_posterior_samples):
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    
    inference = SNPE_C(prior=prior)
    posterior = inference.build_posterior(density_estimator)
    thetas, xs = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=num_sbc_samples, num_workers=11)
    
    ranks, dap_samples = run_sbc(thetas, xs, posterior, num_posterior_samples=num_posterior_samples, num_workers=11)
    results = {"ranks": ranks, "dap_samples": dap_samples, "thetas": thetas, "num_sbc_samples": num_sbc_samples, "num_posterior_samples": num_posterior_samples}

    return results

def tarp(prior, simulator, density_estimator, num_tarp_samples, num_posterior_samples):
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    
    inference = SNPE_C(prior=prior)
    posterior = inference.build_posterior(density_estimator)
    thetas, xs = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=num_tarp_samples, num_workers=11)

    ecp, alpha = run_tarp(thetas, xs, posterior, references=None, num_posterior_samples=num_posterior_samples, use_batched_sampling=False)
    results = {"ecp": ecp, "alpha": alpha, "thetas": thetas, "num_tarp_samples": num_tarp_samples, "num_posterior_samples": num_posterior_samples}

    return results

if __name__ == "__main__":
    prior = get_prior()
    simulator = create_simulator()
    density_estimator = load_model(os.path.join("results", "inference", "trained_model_15.pkl"))
    
    #num_sbc_samples = 1000
    num_tarp_samples = 1000
    num_posterior_samples = 5000
    #sbc_results = sbc(prior, simulator, density_estimator, num_sbc_samples, num_posterior_samples)
    tarp_results = tarp(prior, simulator, density_estimator, num_tarp_samples, num_posterior_samples)

    #torch.save(sbc_results, SBI_CONFIG["sbc_save_path"])
    torch.save(tarp_results, SBI_CONFIG["tarp_save_path"])