import numpy as np
import torch
from src.simulation.config import PATHS, PARAMS
from src.simulation.utils import load_data

def generate_realizations(base_spectra, base_params):
    """Genera realizaciones estadísticas para cada simulación base""" 
    num_realizations = PARAMS["realizations"]["num_realizations"]
    all_spectra = []
    all_params = []
    
    print(f"\nGenerando {num_realizations} realizaciones por simulación...")
    for i, (spectrum, params) in enumerate(zip(base_spectra, base_params)):
        realizations = []
        for _ in range(num_realizations):
            realization = spectrum.copy() # Solo copia de espectros por ahora
            realizations.append(realization)
            all_params.append(params)
        
        all_spectra.extend(realizations)
        print(f"Realizaciones para simulación {i+1}/{len(base_spectra)} completadas")
    
    return np.array(all_spectra), np.array(all_params)

if __name__ == "__main__":
    base_spectra, base_params = load_data("cosmologies")
    spectra, params = generate_realizations(base_spectra, base_params)
    
    torch.save(torch.from_numpy(spectra), PATHS["realizations"]["spectra"])
    torch.save(torch.from_numpy(params), PATHS["realizations"]["params"])
    print(f"\nRealizaciones guardadas en:")
    print(f"- Espectros: {PATHS['realizations']['spectra']}")
    print(f"- Parámetros: {PATHS['realizations']['params']}")