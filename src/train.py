import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

import numpy as np
import torch
import os
from sbi.analysis import plot_summary
from sbi.inference import SNPE_C, NLE, NPSE, FMPE, BNRE, SMCABC, MNLE
from sbi.neural_nets import posterior_score_nn, posterior_nn, flowmatching_nn, likelihood_nn
from src.embedding import get_embedding
from src.prior import get_prior
from src.config import PATHS
from src.simulator import create_simulator, Cl_XX

def inference_SNPE_C(model="nsf", embedding_net=None):
    """Crea el objeto de inferencia SNPE_C"""
    prior = get_prior()
    if embedding_net is not None:
        embedding_net = get_embedding(embedding_net)
        net_builder = posterior_nn(model=model, embedding_net=embedding_net)
        inference = SNPE_C(prior=prior, density_estimator=net_builder)
    else:
        net_builder = posterior_nn(model=model)
        inference = SNPE_C(prior=prior, density_estimator=net_builder)

    return inference

def inference_NPSE(sde_type="ve", embedding_net=None):
    """Crea el objeto de inferencia NPSE"""
    prior = get_prior()
    if embedding_net is not None:
        embedding_net = get_embedding(embedding_net)
        score_estimator = posterior_score_nn(sde_type=sde_type, embedding_net=embedding_net)
        inference = NPSE(prior=prior, score_estimator=score_estimator)
    else:
        score_estimator = posterior_score_nn(sde_type=sde_type)
        inference = NPSE(prior=prior, score_estimator=score_estimator)

    return inference

def inference_FMPE(model="resnet", embedding_net=None):
    """Crea el objeto de inferencia FMPE"""
    prior = get_prior()
    if embedding_net is not None:
        embedding_net = get_embedding(embedding_net)
        density_estimator = flowmatching_nn(model=model, embedding_net=embedding_net)
        inference = FMPE(prior=prior, density_estimator=density_estimator)
    else:
        density_estimator = flowmatching_nn(model=model)
        inference = FMPE(prior=prior, density_estimator=density_estimator)

    return inference

def inference_NLE(model="nsf", embedding_net=None):
    """Crea el objeto de inferencia NLE"""
    prior = get_prior()
    if embedding_net is not None:
        embedding_net = get_embedding(embedding_net)
        neural_likelihood = likelihood_nn(model=model, embedding_net=embedding_net)
        inference = NLE(prior=prior, likelihood_estimator=neural_likelihood)
    else:
        neural_likelihood = likelihood_nn(model=model)
        inference = NLE(prior=prior, likelihood_estimator=neural_likelihood)

    return inference

def inference_MNLE(model="mnle", embedding_net=None):
    """Crea el objeto de inferencia MNLE"""
    prior = get_prior()
    if embedding_net is not None:
        embedding_net = get_embedding(embedding_net)
        neural_likelihood = likelihood_nn(model=model, embedding_net=embedding_net)
        inference = MNLE(prior=prior, likelihood_estimator=neural_likelihood)
    else:
        neural_likelihood = likelihood_nn(model=model)
        inference = MNLE(prior=prior, likelihood_estimator=neural_likelihood)

    return inference

def train_SNPE_C(theta, x, embedding_net=None):
    """Entrenamiento y validación del modelo SNPE_C"""
    inference = inference_SNPE_C(model="nsf", embedding_net=embedding_net)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    fig, axes = plot_summary(inference, tags=["training_loss", "validation_loss"], figsize=(10, 2))

    return density_estimator, fig, axes

def train_NPSE(theta, x, embedding_net=None):
    """Entrenamiento y validación del modelo NPSE"""
    inference = inference_NPSE(sde_type="ve", embedding_net=embedding_net)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    fig, axes = plot_summary(inference, tags=["training_loss", "validation_loss"], figsize=(10, 2))

    return density_estimator, fig, axes

def train_FMPE(theta, x, embedding_net=None):
    """Entrenamiento y validación del modelo FMPE"""
    inference = inference_FMPE(model="resnet", embedding_net=embedding_net)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()

    return density_estimator

def train_NLE(theta, x, embedding_net=None):
    """Entrenamiento y validación del modelo NLE"""
    inference = inference_NLE(model="nsf", embedding_net=embedding_net)
    inference = inference.append_simulations(theta, x)
    likelihood_estimator = inference.train()

    return likelihood_estimator

def train_MNLE(theta, x, embedding_net=None):
    """Entrenamiento y validación del modelo MNLE"""
    inference = inference_MNLE(model="mnle", embedding_net=embedding_net)
    inference = inference.append_simulations(theta, x)
    likelihood_estimator = inference.train()

    return likelihood_estimator

def ABC(theta, x):
    """Entrenamiento y validación del modelo ABC"""
    prior = get_prior()
    simulator = create_simulator("TT")

    inference = SMCABC(simulator=simulator, prior=prior)
    pass

if __name__ == "__main__":
    simulations = torch.load(os.path.join(PATHS["simulations"],"Cls_TT_noise_binned_100000.pt"), weights_only=True)
    theta, x = simulations["theta"], simulations["cl"]
    print(theta.shape, x.shape)

    density_estimator, fig, axes = train_NPSE(theta, x)
    torch.save(density_estimator.state_dict(), os.path.join(PATHS["models"], "NPSE_TT_noise_binned_100000.pth"))
    plt.savefig(os.path.join(PATHS["summary"], "NPSE_TT_noise_binned_100000.png"), dpi=300, bbox_inches='tight')