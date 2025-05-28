import os

RESULTS_DIR = os.path.join("results")
INFERENCE_DIR = os.path.join(RESULTS_DIR, "inference")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")

SBI_CONFIG = {
    "num_simulations": 20000,
    "training_batch_size": 50,
    "training_epochs": 100,
    "validation_fraction": 0.1,
    "hidden_features": 50,
    "num_transforms": 5,
    "device": "cpu",
    "model_save_path": os.path.join(INFERENCE_DIR, "trained_model_18.pkl"),
    "sample_save_path": os.path.join(SAMPLES_DIR, "mcmc_samples_0.pt"),
    "sbc_save_path": os.path.join(CALIBRATION_DIR, "sbc_1.pt"),
    "tarp_save_path": os.path.join(CALIBRATION_DIR, "tarp_3.pt"),
}

os.makedirs(INFERENCE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(CALIBRATION_DIR, exist_ok=True)