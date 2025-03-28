import numpy as np
import torch
from scipy.stats import chi2
import os

def generate_realizations(base_spectra, base_params, num_realizations=50):
    """Genera realizaciones estadísticas para cada simulación base"""
    all_spectra = []
    all_params = []
    
    print(f"\nGenerando {num_realizations} realizaciones por simulación...")
    for i, (spectrum, params) in enumerate(zip(base_spectra, base_params)):
        realizations = []
        for _ in range(num_realizations):
            # Solo copia el espectro base (hasta ahora)
            realization = spectrum.copy()
            realizations.append(realization)
            all_params.append(params)
        
        all_spectra.extend(realizations)
        print(f"Realizaciones para simulación {i+1}/{len(base_spectra)} completadas")
    
    return np.array(all_spectra), np.array(all_params)

if __name__ == "__main__":
    base_spectra = torch.load("data/raw/base_spectra.pt").numpy()
    base_params = torch.load("data/raw/base_params.pt").numpy()
    
    spectra, params = generate_realizations(base_spectra, base_params)
    
    os.makedirs("data/interim", exist_ok=True)
    torch.save(torch.from_numpy(spectra), "data/interim/realizations_spectra.pt")
    torch.save(torch.from_numpy(params), "data/interim/realizations_params.pt")
    print("\nRealizaciones guardadas en data/interim/")