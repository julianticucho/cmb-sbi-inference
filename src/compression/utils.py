import numpy as np
from sklearn.model_selection import KFold

def prepare_data(spectra, params):
    """Normaliza espectros y parámetros"""
    spectra_mean = spectra.mean(0).float()
    spectra_std = spectra.std(0).float()
    params_mean = params.mean(0)
    params_std = params.std(0)
    
    X = (spectra - spectra_mean) / (spectra_std + 1e-8)
    theta = (params - params_mean) / (params_std + 1e-8)
    
    return X.float(), theta.float(), spectra_mean, spectra_std, params_mean, params_std

def create_kfold_split(X, num_cosmologies, realizations_per_cosmology):
    """Misma lógica de KFold por cosmologías"""
    groups = np.arange(len(X)) // realizations_per_cosmology
    unique_groups = np.unique(groups)
    kf = KFold(n_splits=num_cosmologies)
    return kf.split(unique_groups)