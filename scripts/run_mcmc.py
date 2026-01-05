import os
import sys
from cobaya.run import run
from cobaya import load_samples
import getdist.plots as gdplt
from src.config import PATHS
import numpy as np


def build_info():
    info = {
        "theory": {
            "camb": {
                "extra_args": {
                    "lens_potential_accuracy": 1
                }
            }
        },

        "likelihood": {
            "planck_2018_highl_plik.TT_lite_native": None
        },

        "params": {

            "ombh2": {
                "prior": {"min": 0.02212 - 0.00022 * 5,
                        "max": 0.02212 + 0.00022 * 5},
                "latex": r"\Omega_b h^2"
            },

            "omch2": {
                "prior": {"min": 0.1206 - 0.0021 * 5,
                        "max": 0.1206 + 0.0021 * 5},
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
                "prior": {"min": 0.9626 - 0.0057 * 5,
                        "max": 0.9626 + 0.0057 * 5},
                "latex": r"n_s"
            },

            "theta_MC_100": {
                "derived": lambda cosmomc_theta: 100 * cosmomc_theta,
                "latex": r"100\theta_{\rm MC}"
            },

            "ln_10_10_As": {
                "derived": lambda As: np.log(1e10 * As),
                "latex": r"\ln(10^{10} A_s)"
            },
        },

        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.01,
                "max_tries": 1000000
            }
        },

        "output": "results/chains/test"
    }

    return info

def run_mcmc(info):
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
        ["ombh2", "omch2", "ns", "ln_10_10_As", "theta_MC_100"],
        filled=True
    )

    os.makedirs(PATHS["confidence"], exist_ok=True)
    gdplot.export(
        os.path.join(PATHS["confidence"], "gaussian_likelihood_custom.pdf")
    )

    print("Run finished successfully")


if __name__ == "__main__":
    info = build_info()
    run_mcmc(info)


