import os

DATA_DIR = os.path.join("data")
RESULTS_DIR = os.path.join("results") 

PATHS = {
    "simulations": os.path.join(DATA_DIR, "simulations"),
    "plot_simulations": os.path.join(DATA_DIR, "plots"),
    "models": os.path.join(RESULTS_DIR, "models"),
    "posteriors": os.path.join(RESULTS_DIR, "posteriors"),
    "correlation": os.path.join(RESULTS_DIR, "correlation"),
    "calibration": os.path.join(RESULTS_DIR, "calibration"),
    "confidence": os.path.join(RESULTS_DIR, "confidence")
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
