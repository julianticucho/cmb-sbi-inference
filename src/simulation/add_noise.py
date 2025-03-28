import numpy as np
import torch
import os

def add_instrumental_noise(spectra, theta_fwhm=5.0, sigma_T=33.0):
    """Añade ruido instrumental a los espectros"""
    lmax = spectra.shape[1]
    ell = np.arange(lmax)
    theta_fwhm_rad = theta_fwhm * np.pi / (180 * 60)
    
    Nl_TT = (theta_fwhm_rad * sigma_T)**2 * np.exp(ell*(ell+1) * (theta_fwhm_rad**2)/(8*np.log(2)))
    
    return spectra + Nl_TT

def sample_observed_spectra(spectra, f_sky=0.7, l_transition=52):
    """Muestrea espectros observados considerando cobertura parcial del cielo"""
    noisy_spectra = np.zeros_like(spectra)
    lmax = spectra.shape[1]
    
    for i in range(spectra.shape[0]):
        for ell in range(2, lmax):
            C_ell = spectra[i, ell]
            
            if ell < l_transition:
                dof = int(round(f_sky * (2*ell + 1)))
                if dof < 1:
                    dof = 1
                samples = np.random.normal(scale=np.sqrt(C_ell), size=dof)
                noisy_spectra[i, ell] = np.sum(samples**2) / dof
            else:
                var = 2 * C_ell**2 / (f_sky * (2*ell + 1))
                noisy_spectra[i, ell] = np.random.normal(loc=C_ell, scale=np.sqrt(var))
    
    return noisy_spectra

if __name__ == "__main__":
    noise_config = {
        'theta_fwhm': 5.0,     
        'sigma_T': 33.0,      
        'f_sky': 0.7,          
        'l_transition': 52     
    }
    
    spectra = torch.load("data/interim/realizations_spectra.pt").numpy()
    params = torch.load("data/interim/realizations_params.pt").numpy()
    
    print("\nAñadiendo ruido instrumental...")
    noisy_spectra = add_instrumental_noise(
        spectra,
        theta_fwhm=noise_config['theta_fwhm'],
        sigma_T=noise_config['sigma_T']
    )
    
    print("Muestreando espectros observados...")
    observed_spectra = sample_observed_spectra(
        noisy_spectra,
        f_sky=noise_config['f_sky'],
        l_transition=noise_config['l_transition']
    )
    
    os.makedirs("data/noise", exist_ok=True)
    torch.save(torch.from_numpy(observed_spectra), "data/noise/cmb_observed_TTnoise.pt")
    torch.save(torch.from_numpy(params), "data/noise/sim_params.pt")
    
    print("\nResultados finales guardados en data/noise/")
    print(f"- cmb_observed_TTnoise.pt: Espectros con ruido ({observed_spectra.shape})")
    print(f"- sim_params.pt: Parámetros cosmológicos ({params.shape})")