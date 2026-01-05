import torch
import os
import sys
import getdist.plots as gdplt
import numpy as np
from cobaya.run import run
from cobaya import load_samples
from cobaya.likelihood import Likelihood
from src.data import Dataset
from src.processor import Processor
from src.config import PATHS

class PlanckTTGaussian(Likelihood):
    """
    Gaussian likelihood for binned high-ell TT spectra
    """
    def initialize(self):
        dataset = Dataset()
        _, _, _, _, _, lmin, lmax, _, cov = dataset.import_data()

        self.proc = Processor(type_str="TT")
        true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
        self.simulator = self.proc.create_simulator()
        x_obs = self.simulator(true_parameter1)
        x_obs = x_obs[30:2478]
        x_obs = self.proc.bin_high_ell(x_obs, lmin, lmax)
        x_obs = self.proc.add_cov_noise(x_obs, cov, seed=0)

        self.x_obs = x_obs
        self.icov = torch.linalg.inv(cov)

        self.lmin = lmin
        self.lmax = lmax
        self.lmax_global = int(torch.max(lmax).item())

    def get_requirements(self):
        return {
            "Cl": {
                "tt": 2500
            }
        }

    def logp(self, **params):
        try:
            tt = params["Cl"]["tt"]
            print("cl de camb:", tt.shape)
            tt = torch.as_tensor(tt)
            tt = tt[30:2478]
            print("cl de camb reducido:", tt.shape)
            tt = self.proc.bin_high_ell(tt, self.lmin, self.lmax)
            print("cl binned:", tt.shape)
            delta = self.x_obs - tt
            loglike = -0.5 * delta @ self.icov @ delta

            return loglike.item()

        except Exception as e:
            return -1e30
        
def build_info2():
    info = {
        "theory": {
            "camb": {
                "extra_args": {
                    "lens_potential_accuracy": 1,
                    "AccuracyBoost": 1.0,
                    "lmax": 2500,
                    "nonlinear": "both",
                }
            }
        },

        "likelihood": {
            'my_likelihood': {
                "external": PlanckTTGaussian
            },
        },

        "params": {
            "tau": {
                "value": 0.0522,
                "latex": r"\tau"
            },
            "ombh2": {
                "prior": {
                    "min": 0.02212 - 5 * 0.00022,
                    "max": 0.02212 + 5 * 0.00022
                },
                "latex": r"\Omega_b h^2",
            },
            "omch2": {
                "prior": {
                    "min": 0.1206 - 5 * 0.0021,
                    "max": 0.1206 + 5 * 0.0021
                },
                "latex": r"\Omega_c h^2"
            },
            "cosmomc_theta": {
                "prior": {
                    "min": 1.04077 / 100 - 0.00047 / 100 * 5,
                    "max": 1.04077 / 100 + 0.00047 / 100 * 5
                },
                "latex": r"\theta_{\rm MC}"
            },
            "As": {
                "prior": {
                    "min": np.exp(3.04 - 5 * 0.016) / 1e10,
                    "max": np.exp(3.04 + 5 * 0.016) / 1e10,
                },
                "latex": r"A_s"
            },
            "ns": {
                "prior": {
                    "min": 0.9626 - 5 * 0.0057,
                    "max": 0.9626 + 5 * 0.0057
                },
                "latex": r"n_s"
            },
            "theta_MC_100": {
                "derived": lambda cosmomc_theta: 100 * cosmomc_theta,
                "latex": r"100\theta_{\rm MC}"
            },
            "ln_10_10_As": {
                "derived": lambda As: np.log(1e10 * As),
                "latex": r"\ln(10^{10} A_s)"
            }
        },

        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.01,
                "max_tries": 1_000_000
            }
        },
        "output": "results/chains/gaussian_likelihood_custom_V2"
    }

    return info


def run_mcmc2(info):
    for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
        if k in sys.argv:
            info[v] = True

    updated_info, sampler = run(info)

    gd_sample = load_samples(
        info["output"],
        to_getdist=True
    )

    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(
        gd_sample,
        ["ombh2", "omch2", "theta_MC_100", "ln_10_10_As", "ns"],
        filled=True
    )

    os.makedirs(PATHS["confidence"], exist_ok=True)
    gdplot.export(
        os.path.join(PATHS["confidence"], "gaussian_likelihood_custom_V2.pdf")
    )
    print("Run finished successfully")

if __name__ == "__main__":
    info = build_info2()
    run_mcmc2(info)