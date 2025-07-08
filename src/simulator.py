import camb
import numpy as np
import torch
import os
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.inference import simulate_for_sbi
from src.config import PARAMS, PATHS
from src.prior import get_prior
from src.config import PATHS
from src.bin import bin_simulations

def compute_spectrum(params): 
    """Calcula el espectro teórico CMB para un conjunto de parámetros"""
    ombh2, omch2, theta_MC_100, ln_10_10_As, ns = params
    pars = camb.CAMBparams()
    pars.set_cosmology(                
        ombh2=ombh2,            
        omch2=omch2,     
        tau=0.0522,
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

        elif type_str == "TT+noise":
            cmb_power_spectra = compute_spectrum(theta)[:2551]
            cmb_power_spectra = add_instrumental_noise(cmb_power_spectra)
            cmb_power_spectra = sample_observed_spectra(cmb_power_spectra)

        elif type_str == "TT_binned":
            cmb_power_spectra = compute_spectrum(theta)[:2551]
            cmb_power_spectra = torch.from_numpy(cmb_power_spectra).float()
            cmb_power_spectra = cmb_power_spectra.unsqueeze(0)
            cmb_power_spectra = bin_simulations(cmb_power_spectra, 0, 2550, 500)[1]
            cmb_power_spectra = cmb_power_spectra.squeeze(0).numpy()

        elif type_str == "EE_binned":
            cmb_power_spectra = compute_spectrum(theta)[2551:5102]
            cmb_power_spectra = torch.from_numpy(cmb_power_spectra).float()
            cmb_power_spectra = cmb_power_spectra.unsqueeze(0)
            cmb_power_spectra = bin_simulations(cmb_power_spectra, 0, 2550, 500)[1]
            cmb_power_spectra = cmb_power_spectra.squeeze(0).numpy()

        elif type_str == "BB_binned":
            cmb_power_spectra = compute_spectrum(theta)[5102:7653]
            cmb_power_spectra = torch.from_numpy(cmb_power_spectra).float()
            cmb_power_spectra = cmb_power_spectra.unsqueeze(0)
            cmb_power_spectra = bin_simulations(cmb_power_spectra, 0, 2550, 500)[1]
            cmb_power_spectra = cmb_power_spectra.squeeze(0).numpy()

        elif type_str == "TE_binned":
            cmb_power_spectra = compute_spectrum(theta)[7653:]
            cmb_power_spectra = torch.from_numpy(cmb_power_spectra).float()
            cmb_power_spectra = cmb_power_spectra.unsqueeze(0)
            cmb_power_spectra = bin_simulations(cmb_power_spectra, 0, 2550, 500)[1]
            cmb_power_spectra = cmb_power_spectra.squeeze(0).numpy()
            
        elif type_str == "TT+EE+TE_binned":
            TT = compute_spectrum(theta)[:2551]
            TT = torch.from_numpy(TT).float()
            TT = TT.unsqueeze(0)
            TT = bin_simulations(TT, 0, 2550, 500)[1]
            TT = TT.squeeze(0).numpy()

            EE = compute_spectrum(theta)[2551:5102]
            EE = torch.from_numpy(EE).float()
            EE = EE.unsqueeze(0)
            EE = bin_simulations(EE, 0, 2550, 500)[1]
            EE = EE.squeeze(0).numpy()

            TE = compute_spectrum(theta)[7653:]
            TE = torch.from_numpy(TE).float()
            TE = TE.unsqueeze(0)
            TE = bin_simulations(TE, 0, 2550, 500)[1]
            TE = TE.squeeze(0).numpy()

            cmb_power_spectra = np.concatenate((TT, EE, TE))

        elif type_str == "TT+EE+BB+TE_binned":
            TT = compute_spectrum(theta)[:2551]
            TT = torch.from_numpy(TT).float()
            TT = TT.unsqueeze(0)
            TT = bin_simulations(TT, 0, 2550, 500)[1]
            TT = TT.squeeze(0).numpy()

            EE = compute_spectrum(theta)[2551:5102]
            EE = torch.from_numpy(EE).float()
            EE = EE.unsqueeze(0)
            EE = bin_simulations(EE, 0, 2550, 500)[1]
            EE = EE.squeeze(0).numpy()

            BB = compute_spectrum(theta)[5102:7653]
            BB = torch.from_numpy(BB).float()
            BB = BB.unsqueeze(0)
            BB = bin_simulations(BB, 0, 2550, 500)[1]
            BB = BB.squeeze(0).numpy()

            TE = compute_spectrum(theta)[7653:]
            TE = torch.from_numpy(TE).float()
            TE = TE.unsqueeze(0)
            TE = bin_simulations(TE, 0, 2550, 500)[1]
            TE = TE.squeeze(0).numpy()

            cmb_power_spectra = np.concatenate((TT, EE, BB, TE), axis=0)
        
        elif type_str == "TT+noise+binned":
            cmb_power_spectra = compute_spectrum(theta)[:2551]
            cmb_power_spectra = add_instrumental_noise(cmb_power_spectra)
            cmb_power_spectra = sample_observed_spectra(cmb_power_spectra)

            cmb_power_spectra = torch.from_numpy(cmb_power_spectra).float()
            cmb_power_spectra = cmb_power_spectra.unsqueeze(0)
            cmb_power_spectra = bin_simulations(cmb_power_spectra, 0, 2550, 500)[1]
            cmb_power_spectra = cmb_power_spectra.squeeze(0).numpy()

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

def generate_noise(x):
    """Añade ruido a los Cls a partir de un tensor x.shape: (num_simulations, 4*2551)"""
    x_np = x.numpy() if torch.is_tensor(x) else x
    spectra_with_noise = np.array([add_instrumental_noise(spec) for spec in x_np])
    noisy_spectra = np.array([sample_observed_spectra(spec) for spec in spectra_with_noise])

    return torch.from_numpy(noisy_spectra).float()

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
    theta, x = generate_cosmologies(num_simulations=25000)
    tensor_dict = {"theta": theta, "x": x}
    print(theta.shape, x.shape)
    torch.save(tensor_dict, os.path.join(PATHS["simulations"], "all_Cls_tau_25000.pt"))
    print(f"Simulaciones completadas")

    # simulations = torch.load(os.path.join(PATHS["simulations"], "all_Cls_100000.pt"), weights_only=True)
    # theta, x = simulations["theta"], simulations["x"]
    # x_noise = generate_noise(Cl_XX(x, "TT"))

    # tensor_dict = {"theta": theta, "x": x_noise}
    # torch.save(tensor_dict, os.path.join(PATHS["simulations"], "Cls_TT_noise_100000.pt"))
    # print(f"Simulaciones completadas")


