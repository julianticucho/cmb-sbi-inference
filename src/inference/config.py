import os

RESULTS_DIR = os.path.join("results", "inference")

EMBEDDING_CONFIG = {
    "input_shape": (2401,),
    "in_channels": 1,
    "out_channels_per_layer": [16, 32, 64],
    "num_conv_layers": 3,
    "num_linear_layers": 2,
    "output_dim": 8,
    "kernel_size": 5,
    "pool_kernel_size": 4
}

SBI_CONFIG = {
    "num_simulations": 100,
    "training_batch_size": 5,
    "training_epochs": 100,
    "validation_fraction": 0.1,
    "hidden_features": 50,
    "num_transforms": 5,
    "device": "cpu",
    "model_save_path": os.path.join(RESULTS_DIR, "trained_model_1.pkl")
}

PARAM_RANGES = {
    'Omega_m': (0.1, 0.7),
    'Omega_b': (0.04, 0.06),
    'h': (0.6, 0.8),
    'sigma_8': (0.6, 1.1),
    'ns': (0.9, 1.0),
    'tau': (0.04, 0.08)
}

os.makedirs(RESULTS_DIR, exist_ok=True)