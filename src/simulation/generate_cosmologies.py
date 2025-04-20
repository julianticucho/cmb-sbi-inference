import camb
import torch
import numpy as np
from src.simulation.config import PATHS, PARAMS

def generate_cosmological_parameters(num_sims, param_ranges):
    """Genera parámetros cosmológicos aleatorios dentro de los rangos especificados"""
    params = torch.zeros(num_sims, len(param_ranges))
    for i, (key, (min_val, max_val)) in enumerate(param_ranges.items()):
        params[:, i] = torch.FloatTensor(num_sims).uniform_(min_val, max_val)
    
    return params

def compute_spectrum(params):
    """Calcula el espectro teórico CMB para un conjunto de parámetros"""
    Omega_m, Omega_b, h, sigma_8, ns, tau = params
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=h*100,                   
        ombh2=Omega_b * h**2,            
        omch2=(Omega_m - Omega_b) * h**2,     
        tau=tau                     
    )
    pars.InitPower.set_params(
        ns=ns,                      
        As=2e-9 * (sigma_8/0.8)**2      
    )
    pars.set_accuracy(AccuracyBoost=1.0)  
    pars.NonLinear = camb.model.NonLinear_both  
    pars.WantLensing = True      
    results = camb.get_results(pars)
    
    return results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:, 0]

def generate_cosmologies():
    """Genera simulaciones con diferentes parámetros cosmológicos"""
    config = PARAMS["cosmologies"]
    params = generate_cosmological_parameters(config["num_sims"], config["param_ranges"])
    spectra = []
    
    print(f"Generando {config['num_sims']} simulaciones cosmológicas base...")
    for i in range(config["num_sims"]):
        spectrum = compute_spectrum(params[i])
        spectra.append(spectrum[:config["lmax"]])
        print(f"Simulación {i+1}/{config['num_sims']} completada")
    
    torch.save(torch.from_numpy(np.array(spectra)), PATHS["cosmologies"]["spectra"])
    torch.save(params, PATHS["cosmologies"]["params"])
    print(f"\nSimulaciones base guardadas en:")
    print(f"- Espectros: {PATHS['cosmologies']['spectra']}")
    print(f"- Parámetros: {PATHS['cosmologies']['params']}")

if __name__ == "__main__":
    generate_cosmologies()