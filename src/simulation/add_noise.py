import numpy as np
import torch
from src.simulation.config import PATHS, PARAMS

def add_instrumental_noise(spectra):
    """Añade ruido instrumental a los espectros"""
    noise_config = PARAMS["noise"]
    lmax = spectra.shape[1]
    ell = np.arange(lmax)
    theta_fwhm_rad = noise_config['theta_fwhm'] * np.pi / (180 * 60)
    
    Nl_TT = (theta_fwhm_rad * noise_config['sigma_T'])**2 * np.exp(ell*(ell+1) * (theta_fwhm_rad**2)/(8*np.log(2)))
    
    return spectra + Nl_TT

def sample_observed_spectra(spectra):
    """Muestrea espectros observados considerando cobertura parcial del cielo"""
    noise_config = PARAMS["noise"]
    noisy_spectra = np.zeros_like(spectra)
    lmax = spectra.shape[1]
    
    for i in range(spectra.shape[0]):
        for ell in range(2, lmax):
            C_ell = spectra[i, ell]
            
            if ell < noise_config['l_transition']:
                dof = int(round(noise_config['f_sky'] * (2*ell + 1)))
                if dof < 1:
                    dof = 1
                samples = np.random.normal(scale=np.sqrt(C_ell), size=dof)
                noisy_spectra[i, ell] = np.sum(samples**2) / dof
            else:
                var = 2 * C_ell**2 / (noise_config['f_sky'] * (2*ell + 1))
                noisy_spectra[i, ell] = np.random.normal(loc=C_ell, scale=np.sqrt(var))
    
    return noisy_spectra

if __name__ == "__main__":
    spectra = torch.load(PATHS["realizations"]["spectra"]).numpy()
    params = torch.load(PATHS["realizations"]["params"]).numpy()
    
    print("\nAñadiendo ruido instrumental...")
    noisy_spectra = add_instrumental_noise(spectra)
    
    print("Muestreando espectros observados...")
    observed_spectra = sample_observed_spectra(noisy_spectra)
    
    torch.save(torch.from_numpy(observed_spectra), PATHS["noise"]["spectra"])
    torch.save(torch.from_numpy(params), PATHS["noise"]["params"])
    
    print("\nResultados finales guardados en:")
    print(f"- Espectros con ruido: {PATHS['noise']['spectra']} ({observed_spectra.shape})")
    print(f"- Parámetros cosmológicos: {PATHS['noise']['params']} ({params.shape})")