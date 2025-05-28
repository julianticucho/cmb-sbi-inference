import camb
import numpy as np
import torch
import os
from src.simulator.config import PARAMS, PATHS
from src.simulator.prior import get_prior
from src.simulator.utils import create_directories

def compute_spectrum(params): 
    """Calcula el espectro teórico CMB para un conjunto de parámetros"""
    ombh2, omch2, theta_MC_100, ln_10_10_As, ns = params
    pars = camb.CAMBparams()
    pars.set_cosmology(                
        ombh2=ombh2,            
        omch2=omch2,     
        tau= 0.0522,
        cosmomc_theta=theta_MC_100/100,   
    )
    pars.InitPower.set_params(
        ns=ns,                      
        As=np.exp(ln_10_10_As)/1e10      
    )
    pars.set_for_lmax(2500)
    pars.set_accuracy(AccuracyBoost=1.0)  
    pars.NonLinear = camb.model.NonLinear_both  
    pars.WantLensing = True      
    results = camb.get_results(pars)
    
    return results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:, 0]

def add_instrumental_noise(spectra):
    """Añade ruido instrumental a los espectros"""
    noise_config = PARAMS["noise"]
    lmax = spectra.shape[0]
    ell = np.arange(lmax)
    theta_fwhm_rad = noise_config['theta_fwhm'] * np.pi / (180 * 60)
    
    Nl_TT = (theta_fwhm_rad * noise_config['sigma_T'])**2 * np.exp(ell*(ell+1)*(theta_fwhm_rad**2)/(8*np.log(2)))
    
    return spectra + Nl_TT

def sample_observed_spectra(spectra):
    """Muestrea espectros observados considerando cobertura parcial del cielo"""
    noise_config = PARAMS["noise"]
    noisy_spectra = np.zeros_like(spectra)
    lmax = spectra.shape[0]
    
    for ell in range(2, lmax):
        C_ell = spectra[ell]
        
        if ell < noise_config['l_transition']:
            dof = int(round(noise_config['f_sky'] * (2*ell + 1)))
            if dof < 1:
                dof = 1
            samples = np.random.normal(scale=np.sqrt(C_ell), size=dof)
            noisy_spectra[ell] = np.sum(samples**2) / dof
        else:
            var = 2 * C_ell**2 / (noise_config['f_sky'] * (2*ell + 1))
            noisy_spectra[ell] = np.random.normal(loc=C_ell, scale=np.sqrt(var))
    
    return noisy_spectra

def create_simulator():
    """Crea el simulador siguiendo la pipeline"""
    def simulator(theta):
        base_spectra = compute_spectrum(theta)
        # noisy_spectra = add_instrumental_noise(base_spectra)
        # observed_spectra = sample_observed_spectra(noisy_spectra)
        return torch.from_numpy(base_spectra)

    return simulator

def generate_cosmologies(num_sims):
    """Genera simulaciones con diferentes parámetros cosmológicos"""
    prior = get_prior()
    params = prior.sample((num_sims,))
    spectra = []
    
    print(f"Generando {num_sims} simulaciones cosmológicas base...")
    for theta in params:
        spectrum = compute_spectrum(theta)
        spectra.append(spectrum)
        print(f"Simulación completada")
    
    create_directories()
    torch.save(torch.from_numpy(np.array(spectra)), os.path.join(PATHS["examples"], "spectra.pt")) 
    torch.save(params, os.path.join(PATHS["examples"], "params.pt"))   
    print(f"\nSimulaciones base guardadas en {PATHS['examples']}")

if __name__ == "__main__":
    generate_cosmologies(10)
