import os

BASE_DIR = "data"
FIRST_DATA_DIR = os.path.join(BASE_DIR, "test")

PATHS = {
    "cosmologies": {
        "spectra": os.path.join(FIRST_DATA_DIR, "cosmologies_spectra.pt"),
        "params": os.path.join(FIRST_DATA_DIR, "cosmologies_params.pt")
    },
    
    "realizations": {
        "spectra": os.path.join(FIRST_DATA_DIR, "realizations_spectra.pt"),
        "params": os.path.join(FIRST_DATA_DIR, "realizations_params.pt")
    },
    
    "noise": {
        "spectra": os.path.join(FIRST_DATA_DIR, "noise_spectra.pt"),
        "params": os.path.join(FIRST_DATA_DIR, "noise_params.pt")
    }
}

PARAMS = {
    "cosmologies": {
        "num_sims": 100,
        "lmax": 2500,
        "param_ranges": {
            'Omega_m': (0.1, 0.7),
            'Omega_b': (0.04, 0.06),
            'h': (0.6, 0.8),
            'sigma_8': (0.6, 1.1),
            'ns': (0.9, 1.0),
            'tau': (0.04, 0.08)
        }
    },
    "realizations": {
        "num_realizations": 50
    },
    "noise": {
        'theta_fwhm': 5.0,
        'sigma_T': 33.0,
        'f_sky': 0.7,
        'l_transition': 52
    }
}

def create_directories():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(FIRST_DATA_DIR, exist_ok=True)

create_directories()