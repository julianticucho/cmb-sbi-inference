import torch
import numpy as np

def preprocess_spectra(spectra: np.ndarray) -> torch.Tensor: # No hace nada por ahora
    """Preprocesa los espectros ya simulados antes de entrenarlos/validarlos en el sbi"""
    return torch.from_numpy(spectra).float()

def save_model(model, path: str) -> None:
    """Guarda el modelo en el path especificado"""
    torch.save(model, path)

def load_model(path: str):
    """Carga el modelo guardado en el path especificado"""
    return torch.load(path)