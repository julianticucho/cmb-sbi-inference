import os

DATA_DIR = os.path.join("data")
RESULTS_DIR = os.path.join("results") 

PATHS = {
    "simulations": os.path.join(DATA_DIR, "simulations"),
    "plot_simulations": os.path.join(DATA_DIR, "plots"),
    "planck": os.path.join(DATA_DIR, "planck"),
    "models": os.path.join(RESULTS_DIR, "models"),
    "posteriors": os.path.join(RESULTS_DIR, "posteriors"),
    "correlation": os.path.join(RESULTS_DIR, "correlation"),
    "calibration": os.path.join(RESULTS_DIR, "calibration"),
    "confidence": os.path.join(RESULTS_DIR, "confidence"),
    "summary": os.path.join(RESULTS_DIR, "summary"),
    "consistency": os.path.join(RESULTS_DIR, "consistency"),
    "chains": os.path.join(RESULTS_DIR, "chains"),
}

PARAM_RANGES = {
    'ombh2': (0.02212-0.00022*(5), 0.02212+0.00022*(5)), # 'ombh2': (0.01992, 0.02432),
    'omch2': (0.1206-0.0021*(5), 0.1206+0.0021*(5)), # 'omch2': (0.0996, 0.1416),
    'theta_MC_100': (1.04077-0.00047*(5), 1.04077+0.00047*(5)), # 'theta_MC_100': (1.03607, 1.04547),
    'ln_10_10_As': (3.04-0.016*(5), 3.04+0.016*(5)), # 'ln_10_10_As': (2.88, 3.2),
    'ns': (0.9626-0.0057*(5), 0.9626+0.0057*(5)), # 'ns': (0.9056, 1.0196)
    # 'tau': (0, 0.0522+0.008*10), ## 'tau': (0, 0.0522+0.008*10),
}

PARAMS = {
    "noise": {
        'theta_fwhm': 5.0,
        'sigma_T': 0.0, # 33.0
        'f_sky': 10, # 0.7 - 10
        'l_transition': 1 # 52
    },
    "binning": {
        'bin_width': 500
    }
}

CONFIG = {
    # "limits": [
    #     [0.02212-0.00022*(1/5), 0.02212+0.00022*(1/5)],    
    #     [0.1206-0.0021*(1/5), 0.1206+0.0021*(1/5)],  
    #     [1.04077-0.00047*(1/5), 1.04077+0.00047*(1/5)],      
    #     [3.04-0.016*(1/5), 3.04+0.016*(1/5)],    
    #     [0.9626-0.0057*(1/5), 0.9626+0.0057*(1/5)]
    # ],
    "limits": [
        [0.022068-0.00022*(5), 0.022068+0.00022*(5)],    
        [0.12029-0.0021*(5), 0.12029+0.0021*(5)],  
        [1.04122-0.00047*(5), 1.04122+0.00047*(5)],      
        [3.098-0.016*(5), 3.098+0.016*(5)],    
        [0.9624-0.0057*(5), 0.9624+0.0057*(5)]
    ],
    "param_names": ['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
    "param_labels": [r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$']
}