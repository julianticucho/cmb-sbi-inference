import camb
import numpy as np
import torch
from camb import constants
import os

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

def generate_base_simulations(num_sims=100, lmax=2500):
    """Genera simulaciones base con diferentes parámetros cosmológicos"""
    param_ranges = {
        'Omega_m': (0.1, 0.7),    
        'Omega_b': (0.04, 0.06),  
        'h': (0.6, 0.8),     
        'sigma_8': (0.6, 1.1),    
        'ns': (0.9, 1.0),    
        'tau': (0.04, 0.08)  
    }
    
    params = generate_cosmological_parameters(num_sims, param_ranges)
    spectra = []
    
    print(f"Generando {num_sims} simulaciones cosmológicas base...")
    for i in range(num_sims):
        spectrum = compute_spectrum(params[i])
        spectra.append(spectrum[:lmax])
        print(f"Simulación {i+1}/{num_sims} completada")
    
    os.makedirs("data/raw", exist_ok=True)
    torch.save(torch.from_numpy(np.array(spectra)), "data/raw/base_spectra.pt")
    torch.save(params, "data/raw/base_params.pt")
    print("\nSimulaciones base guardadas en data/raw/")

if __name__ == "__main__":
    generate_base_simulations()