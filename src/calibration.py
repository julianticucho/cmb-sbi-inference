import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

import os
import torch
from sbi.diagnostics import run_sbc, run_tarp
from sbi.analysis.plot import plot_tarp
from src.simulator import generate_cosmologies
from src.config import PATHS
from src.posterior import posterior_SNPE_C, posterior_NPSE

def sbc(posterior, num_sbc_samples=200, num_posterior_samples=1000):
    thetas, xs = generate_cosmologies(num_sbc_samples)
    ranks, dap_samples = run_sbc(thetas, xs, posterior, num_posterior_samples=num_posterior_samples, num_workers=11)
    results = {"ranks": ranks, "dap_samples": dap_samples, "thetas": thetas, "num_sbc_samples": num_sbc_samples, "num_posterior_samples": num_posterior_samples}

    return results

def tarp(posterior, num_tarp_samples=200, num_posterior_samples=1000):
    thetas, xs = generate_cosmologies(num_tarp_samples)
    ecp, alpha = run_tarp(thetas, xs, posterior, references=None, num_posterior_samples=num_posterior_samples, use_batched_sampling=False)
    fig, axes = plot_tarp(ecp, alpha, title="TARP diagnostic")

    return fig, axes

if __name__ == "__main__":  
    simulations = torch.load(os.path.join(PATHS["simulations"],"simulations_25000.pt"), weights_only=True)
    theta, x = simulations["theta"], simulations["x"]
    posterior = posterior_NPSE(os.path.join(PATHS["models"],"NPSE_25000.pth"), theta, x)

    fig, axes = tarp(posterior, num_tarp_samples=1000, num_posterior_samples=1000)
    plt.savefig(os.path.join(PATHS["calibration"], "NPSE_25000.png"))