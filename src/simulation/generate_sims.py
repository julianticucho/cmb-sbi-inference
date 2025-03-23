import camb
import numpy as np
import torch
from camb import constants

def generate_Cl(Omega_m=0.3, Omega_b=0.049, h=0.6727, sigma_8=0.8, ns=0.96, tau=0.06):
    """
    Genera espectros de potencia angular CMB con los 6 parámetros cosmológicos principales.
    
    Parámetros:
    - Omega_m: Densidad total de materia (materia oscura + bariónica)
    - Omega_b: Densidad de materia bariónica
    - h: Parámetro de Hubble (H0 = h * 100 km/s/Mpc)
    - sigma_8: Amplitud de fluctuaciones de materia
    - ns: Índice espectral
    - tau: Profundidad óptica de reionización
    """
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
    Cl_total = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']

    return torch.tensor(Cl_total[:3000, :4])  

if __name__ == "__main__":
    param_ranges = {
        'Omega_m': (0.1, 0.7),    
        'Omega_b': (0.04, 0.06),  
        'h': (0.6, 0.8),     
        'sigma_8': (0.6, 1.1),    
        'ns': (0.9, 1.0),    
        'tau': (0.04, 0.08)  
    }
    
    num_sims = 10
    params = torch.zeros(num_sims, len(param_ranges))
    
    for i, (key, (min_val, max_val)) in enumerate(param_ranges.items()):
        params[:, i] = torch.FloatTensor(num_sims).uniform_(min_val, max_val)
    
    Cl_sims = torch.stack([generate_Cl(
        Omega_m=p[0].item(),
        Omega_b=p[1].item(),
        h=p[2].item(),
        sigma_8=p[3].item(),
        ns=p[4].item(),
        tau=p[5].item()
    ) for p in params])
    
    torch.save(Cl_sims, "data/raw/cmb_angular_sims.pt")
    torch.save(params, "data/raw/angular_sim_params.pt")
    print(f"Simulaciones de espectros de potencia angulares guardadas ({num_sims} realizaciones)")