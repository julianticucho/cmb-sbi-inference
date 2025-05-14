import os

RESULTS_DIR = os.path.join("results", "inference")

SBI_CONFIG = {
    "num_simulations": 20000,
    "training_batch_size": 50,
    "training_epochs": 100,
    "validation_fraction": 0.1,
    "hidden_features": 50,
    "num_transforms": 5,
    "device": "cpu",
    "model_save_path": os.path.join(RESULTS_DIR, "trained_model_18.pkl"),
    "sample_save_path": os.path.join(RESULTS_DIR, "mcmc_samples_0.pt")
}

os.makedirs(RESULTS_DIR, exist_ok=True)