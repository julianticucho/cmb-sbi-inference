import torch
from sklearn.preprocessing import StandardScaler
from src.inference.config import DATA_PATHS, SBI_CONFIG

def load_data() -> tuple:
    data_type = SBI_CONFIG["input_data"]
    spectra = torch.load(DATA_PATHS[data_type]["spectra"]).float()
    params = torch.load(DATA_PATHS[data_type]["params"]).float()
    
    if len(spectra) > SBI_CONFIG["num_simulations"]:
        idx = torch.randperm(len(spectra))[:SBI_CONFIG["num_simulations"]]
        spectra = spectra[idx]
        params = params[idx]
    
    return spectra, params

def preprocess_spectra(spectra: torch.Tensor) -> torch.Tensor:
    spectra = torch.log(spectra + 1e-16)
    scaler = StandardScaler()
    return torch.from_numpy(scaler.fit_transform(spectra.numpy())).float()

def save_model(model, path: str) -> None:
    """Guarda solo lo esencial: el modelo completo"""
    torch.save(model, path)

def load_model(path: str):
    """Carga el modelo completo"""
    return torch.load(path)