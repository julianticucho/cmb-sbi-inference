import os
from torch import Tensor
from sbi.utils import BoxUniform

BASE_DIR = "data"
FIRST_DATA_DIR = os.path.join(BASE_DIR, "raw")
INFERENCE_DIR = os.path.join("results", "inference")

DATA_PATHS = {
    "noise": {
        "spectra": os.path.join(FIRST_DATA_DIR, "noise_spectra.pt"),
        "params": os.path.join(FIRST_DATA_DIR, "noise_params.pt")
    }
}

SBI_CONFIG = {
    "input_data": "noise",
    "num_simulations": 5000,
    "training_batch_size": 100,
    "training_epochs": 100,
    "validation_fraction": 0.1,
    "hidden_features": 50,
    "num_transforms": 5,
    "device": "cpu",
    "density_estimator": "maf",
    "model_save_path": os.path.join(INFERENCE_DIR, "trained_model.pkl")
}

PARAM_RANGES = {
    'Omega_m': (0.1, 0.7),
    'Omega_b': (0.04, 0.06),
    'h': (0.6, 0.8),
    'sigma_8': (0.6, 1.1),
    'ns': (0.9, 1.0),
    'tau': (0.04, 0.08)
}

def get_prior() -> BoxUniform:
    lows = Tensor([v[0] for v in PARAM_RANGES.values()])
    highs = Tensor([v[1] for v in PARAM_RANGES.values()])
    return BoxUniform(low=lows, high=highs)

os.makedirs(INFERENCE_DIR, exist_ok=True)