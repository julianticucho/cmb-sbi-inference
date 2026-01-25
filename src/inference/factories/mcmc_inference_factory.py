import numpy as np
from pathlib import Path
from typing import Any, Dict, Callable, Optional, Tuple
from src.core.paths import Paths
from src.inference.likelihoods.gaussian_planck_tt import GaussianPlanckTTLikelihood


class MCMCInferenceFactory:

    @staticmethod
    def get_available_configurations() -> Dict[str, Callable[..., Tuple[Dict[str, Any], Path]]]:
        return {
            "gaussian_mixture_demo": MCMCInferenceFactory.create_gaussian_mixture_demo,
            "planck_tt_gaussian": MCMCInferenceFactory.create_planck_tt_gaussian,
        }

    @staticmethod
    def get_configuration(config_name: str, **kwargs) -> Tuple[Dict[str, Any], Path]:
        configs = MCMCInferenceFactory.get_available_configurations()
        if config_name not in configs:
            raise ValueError(
                f"Configuration {config_name} not found, available: {list(configs.keys())}"
            )
        return configs[config_name](**kwargs)

    @staticmethod
    def create_gaussian_mixture_demo(
        run_name: str = "cobaya_gaussian_mixture",
        seed: Optional[int] = None,
        mcmc: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        paths = Paths()
        paths.ensure_directories()
        output_prefix = paths.chains_dir / run_name

        info: Dict[str, Any] = {
            "likelihood": {
                "gaussian_mixture": {
                    "means": [0.2, 0],
                    "covs": [[0.1, 0.05], [0.05, 0.2]],
                    "derived": True,
                }
            },
            "params": {
                "a": {"prior": {"min": -0.5, "max": 3}, "latex": r"\\alpha"},
                "b": {
                    "prior": {"dist": "norm", "loc": 0, "scale": 1},
                    "ref": 0,
                    "proposal": 0.5,
                    "latex": r"\\beta",
                },
                "derived_a": {"latex": r"\\alpha'"},
                "derived_b": {"latex": r"\\beta'"},
            },
            "sampler": {"mcmc": mcmc or {}},
            "output": str(output_prefix),
        }

        if seed is not None:
            info["sampler"]["mcmc"]["seed"] = int(seed)

        return info, output_prefix

    @staticmethod
    def create_planck_tt_gaussian(
        run_name: str = "planck_tt_gaussian",
        seed: Optional[int] = None,
        mcmc: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        paths = Paths()
        paths.ensure_directories()
        output_prefix = paths.chains_dir / run_name

        info: Dict[str, Any] = {
            "likelihood": {
                "planck_tt_gaussian": {
                    "external": GaussianPlanckTTLikelihood,
                    "speed": -1,
                }
            },
            "theory": {
                "camb": {
                    "extra_args": {
                        "lens_potential_accuracy": 1,
                        "AccuracyBoost": 1.0,
                        "lmax": 2500,
                        "nonlinear": "both",
                        "tau": 0.0522,
                    }
                }
            },
            "params": {
                "ombh2": {
                    "prior": {"min": 0.02212-0.00022*5, "max": 0.02212+0.00022*5},
                    "latex": r"\Omega_b h^2",
                },
                "omch2": {
                    "prior": {"min": 0.1206-0.0021*5, "max": 0.1206+0.0021*5},
                    "latex": r"\Omega_c h^2",
                },
                "cosmomc_theta": {
                    "prior": {"min": (1.04077-0.00047*5)/100, "max": (1.04077+0.00047*5)/100},
                    "latex": r"\theta_{\rm MC}",
                },
                "As": {
                    "prior": {"min": np.exp(3.04-0.016*5)/1e10, "max": np.exp(3.04+0.016*5)/1e10},
                    "latex": r"A_s",
                },
                "ns": {
                    "prior": {"min": 0.9626-0.0057*5, "max": 0.9626+0.0057*5},
                    "latex": r"n_s",
                },
            },
            "sampler": {"mcmc": mcmc or {}},
            "output": str(output_prefix),
        }

        if seed is not None:
            info["sampler"]["mcmc"]["seed"] = int(seed)

        return info, output_prefix
