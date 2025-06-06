import numpy as np
import torch
import os
from sbi.inference import SNPE_C, NLE, NPSE
from sbi.utils.user_input_checks import process_prior
from src.prior import get_prior
from src.config import PATHS

def train_SNPE_C(theta, x):
    """Entrenamiento y validación del modelo NPE"""
    prior = get_prior()
    prior, _, _ = process_prior(prior)

    inference = SNPE_C(prior=prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()

    return density_estimator

def train_NLE(theta, x):
    """Entrenamiento y validación del modelo NLE"""
    prior = get_prior()
    prior, _, _ = process_prior(prior)

    inference = NLE(prior=prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()

    return density_estimator

def train_NPSE(theta, x):
    """Entrenamiento y validación del modelo NPSE"""
    prior = get_prior()
    prior, _, _ = process_prior(prior)

    inference = NPSE(prior=prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()

    return density_estimator

if __name__ == "__main__":
    simulations = torch.load(os.path.join(PATHS["simulations"],"all_Cls_25000.pt"), weights_only=True)
    theta, x = simulations["theta"], simulations["x"][:,:7653]
    print(theta.shape, x.shape)

    density_estimator = train_NPSE(theta, x)
    torch.save(density_estimator.state_dict(), os.path.join(PATHS["models"], "NPSE_TT_EE_BB_25000.pth"))