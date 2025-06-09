import camb
import numpy as np
import torch
import os
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.inference import simulate_for_sbi
from src.config import PARAMS, PATHS
from src.prior import get_prior
from src.config import PATHS

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
        As=np.exp(np.asarray(ln_10_10_As))/1e10      
    )
    pars.set_for_lmax(2500)
    pars.set_accuracy(AccuracyBoost=1.0)  
    pars.NonLinear = camb.model.NonLinear_both  
    pars.WantLensing = True      
    results = camb.get_results(pars)
    cmb_power_spectra = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    
    return np.concatenate([cmb_power_spectra[:, i] for i in range(4)])
    
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

def create_simulator(type_str="TT+EE+BB+TE"):
    """Crea el simulador siguiendo la pipeline"""
    def simulator(theta):
        if type_str == "TT+EE+BB+TE":
            cmb_power_spectra = compute_spectrum(theta)
        elif type_str == "TT":
            cmb_power_spectra = compute_spectrum(theta)[:2551]
        elif type_str == "EE":
            cmb_power_spectra = compute_spectrum(theta)[2551:5102]
        elif type_str == "BB":
            cmb_power_spectra = compute_spectrum(theta)[5102:7653]
        elif type_str == "TE":
            cmb_power_spectra = compute_spectrum(theta)[7653:]
        elif type_str == "TT+EE":
            cmb_power_spectra = compute_spectrum(theta)[:5102]

        return torch.from_numpy(cmb_power_spectra)

    return simulator

def generate_cosmologies(num_simulations):
    """Genera simulaciones en batches con parámetros cosmológicos dentro del prior"""
    prior = get_prior()
    simulator = create_simulator()

    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
    theta, x = simulate_for_sbi(simulator_wrapper, proposal=prior, num_simulations=num_simulations, num_workers=11, seed=1)

    return theta, x

def Cl_XX(concatenate_batches, spectrum_type):
    """Devuelve un vector 1D de los espectros de dos puntos concatenados"""
    if spectrum_type == "TT":
        return concatenate_batches[:, :2551]
    elif spectrum_type == "EE":
        return concatenate_batches[:, 2551:5102]
    elif spectrum_type == "BB":
        return concatenate_batches[:, 5102:7653]
    elif spectrum_type == "TE":
        return concatenate_batches[:, 7653:]
    elif spectrum_type == "TT+EE":
        return concatenate_batches[:, :5102]
    elif spectrum_type == "TT+EE+BB+TE":
        return concatenate_batches

if __name__ == "__main__":
    theta, x = generate_cosmologies(num_simulations=10)
    tensor_dict = {"theta": theta, "x": x}
    torch.save(tensor_dict, os.path.join(PATHS["simulations"], "all_Cls_25000.pt"))
    print(f"Simulaciones completadas")

