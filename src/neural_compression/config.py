TRAIN_CONFIG = {
    'num_cosmologies': 100,
    'realizations_per_cosmology': 50,
    'optuna_trials': 10,
    'epochs': 200,
    'patience': 10,
    'lmax': 2400  
}

PARAM_NAMES = ['Omega_m', 'Omega_b', 'h', 'sigma_8', 'ns', 'tau']

DATA_PATHS = {
    'spectra': 'data/noise/cmb_observed_TTnoise.pt',
    'params': 'data/noise/sim_params.pt',
    'output': 'results/models/'
}