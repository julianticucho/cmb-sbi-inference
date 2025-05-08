import os

RESULTS_DIR = "results"
DATA_DIR = "data"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

PATHS = {
    "base_spectra": os.path.join(PLOTS_DIR, "base_spectra"),
    "noisy_spectra": os.path.join(PLOTS_DIR, "noisy_spectra"),
    "examples": os.path.join(DATA_DIR, "examples")

}

PARAM_RANGES = {
    'ombh2': (0.01992, 0.02432),
    'omch2': (0.0996, 0.1416),
    'theta_MC_100': (1.03607, 1.04547),
    'ln_10_10_As': (2.88, 3.2),
    'ns': (0.9056, 1.0196)
}

PARAMS = {
    "noise": {
        'theta_fwhm': 5.0,
        'sigma_T': 33.0,
        'f_sky': 0.7,
        'l_transition': 52
    }
}
