import torch
from src.simulation.config import PATHS

def load_data(input_data) -> tuple:
    spectra = torch.load(PATHS[input_data]["spectra"], weights_only=True).numpy()
    params = torch.load(PATHS[input_data]["params"], weights_only=True).numpy()
    
    return spectra, params