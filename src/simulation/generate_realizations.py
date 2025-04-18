import numpy as np
import torch
from src.simulation.config import PATHS, PARAMS

def generate_realizations(base_spectra, base_params):
    """Genera realizaciones estadísticas para cada simulación base"""
    num_realizations = PARAMS["realizations"]["num_realizations"]
    all_spectra = []
    all_params = []
    
    print(f"\nGenerando {num_realizations} realizaciones por simulación...")
    for i, (spectrum, params) in enumerate(zip(base_spectra, base_params)):
        realizations = []
        for _ in range(num_realizations):
            realization = spectrum.copy()
            realizations.append(realization)
            all_params.append(params)
        
        all_spectra.extend(realizations)
        print(f"Realizaciones para simulación {i+1}/{len(base_spectra)} completadas")
    
    return np.array(all_spectra), np.array(all_params)

if __name__ == "__main__":
    base_spectra = torch.load(PATHS["cosmologies"]["spectra"]).numpy()
    base_params = torch.load(PATHS["cosmologies"]["params"]).numpy()
    
    spectra, params = generate_realizations(base_spectra, base_params)
    
    torch.save(torch.from_numpy(spectra), PATHS["realizations"]["spectra"])
    torch.save(torch.from_numpy(params), PATHS["realizations"]["params"])
    print(f"\nRealizaciones guardadas en:")
    print(f"- Espectros: {PATHS['realizations']['spectra']}")
    print(f"- Parámetros: {PATHS['realizations']['params']}")