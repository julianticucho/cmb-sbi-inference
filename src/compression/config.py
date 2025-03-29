TRAIN_CONFIG = {
    'num_cosmologies': 100,
    'realizations_per_cosmology': 50,
    'optuna_trials': 10,      # Para búsqueda de hiperparámetros (opcional)
    'epochs': 200,
    'patience': 10,
    'lmax': 2400,
    'latent_dim': 64,         # Nueva: dimensión del espacio latente
    'learning_rate': 1e-3,
    'batch_size': 128
}

PARAM_NAMES = ['Omega_m', 'Omega_b', 'h', 'sigma_8', 'ns', 'tau']

DATA_PATHS = {
    'spectra': 'data/noise/cmb_observed_TTnoise.pt',  # [5000, 2401]
    'params': 'data/noise/sim_params.pt',             # [5000, 6]
    'output': 'results/models/'
}