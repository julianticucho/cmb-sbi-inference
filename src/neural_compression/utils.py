import numpy as np
import torch
from sklearn.model_selection import KFold

def prepare_data(spectra, params):
    """Normaliza los datos respetando la estructura por cosmologías"""
    spectra = spectra.numpy() if torch.is_tensor(spectra) else spectra
    params = params.numpy() if torch.is_tensor(params) else params
    
    # Calculamos estadísticas solo con el conjunto de entrenamiento para evitar data leakage
    train_mask = np.zeros(len(spectra), dtype=bool)
    train_mask[:400] = True  # Usamos 8 cosmologías (400 realizaciones) para cálculo de estadísticas
    
    spectra_mean = spectra[train_mask].mean(axis=0)
    spectra_std = spectra[train_mask].std(axis=0)
    params_mean = params[train_mask].mean(axis=0)
    params_std = params[train_mask].std(axis=0)
    
    # Normalización
    X = (spectra - spectra_mean) / (spectra_std + 1e-8)
    y = (params - params_mean) / (params_std + 1e-8)
    
    return X, y, spectra_mean, spectra_std, params_mean, params_std

def create_kfold_split(X, num_cosmologies, realizations_per_cosmology):
    """Genera splits que mantienen juntas las realizaciones de cada cosmología"""
    # Creamos grupos por cosmología (cada 50 realizaciones)
    groups = np.arange(len(X)) // realizations_per_cosmology
    unique_groups = np.unique(groups)
    
    # KFold sobre las cosmologías, no sobre realizaciones individuales
    kf = KFold(n_splits=num_cosmologies)
    return kf.split(unique_groups)