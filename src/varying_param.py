import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from src.config import PARAM_RANGES, PATHS
from src.simulator import compute_spectrum, create_simulator, Cl_XX

def plot_varying_param(simulator, output_dir):
    central_params = {k: (v[0] + v[1])/2 for k, v in PARAM_RANGES.items()}
    n_values = 4 
    params_order = ['ombh2', 'omch2', 'theta_MC_100', 'ln_10_10_As', 'ns']
    param_latex = {
        'ombh2': r'$\Omega_b h^2$',
        'omch2': r'$\Omega_c h^2$',
        'theta_MC_100': r'$100\theta_{\rm MC}$',
        'ln_10_10_As': r'$\ln(10^{10}A_s)$',
        'ns': r'$n_s$'
    }
    for param_name in params_order:
        fig, axes = plt.figure(figsize=(12, 8))
        param_values = np.linspace(PARAM_RANGES[param_name][0], PARAM_RANGES[param_name][1], n_values)
        
        for i, val in enumerate(param_values):
            current_params = central_params.copy()
            current_params[param_name] = val
            param_list = [current_params[p] for p in params_order]
            spectrum = simulator(param_list) 
            
            plt.plot(spectrum, label=f'{param_latex[param_name]} = {val:.4f}', alpha=0.7)
        
        plt.title(f'CMB Power Spectrum variando {param_latex[param_name]}', fontsize=14, pad=20)
        plt.xlabel('Multipolo $\ell$')
        plt.ylabel('$C_\ell^{TT}$ [$\mu K^2$]')
        plt.legend()
        
        filename = os.path.join(output_dir, f'varying_{param_name}.png')
        plt.tight_layout()
        plt.savefig(filename)  
        
        print(f'Gráfico guardado: {filename}')
    
def plot_cls_subplots(Cls_TT, Cls_EE, Cls_BB, Cls_TE, n_cosmologies=3, figsize=(16, 6)):
    """Crea una figura con 4 subplots (2x2) mostrando los espectros de potencia angular."""
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('CMB angular power spectrum (scalar modes with lensing)', fontsize=16, y=1.02)
    
    plot_configs = [
        (Cls_TT, '$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu K^2$]', 'TT'),
        (Cls_EE, '$\ell(\ell+1)C_\ell^{EE}/2\pi$ [$\mu K^2$]', 'EE'),
        (Cls_BB, '$\ell(\ell+1)C_\ell^{BB}/2\pi$ [$\mu K^2$]', 'BB'),
        (Cls_TE, '$\ell(\ell+1)C_\ell^{TE}/2\pi$ [$\mu K^2$]', 'TE')
    ]
    
    for ax, (Cls, ylabel, cl_type) in zip(axs.flat, plot_configs):
        Cls_np = Cls.numpy() if hasattr(Cls, 'numpy') else Cls

        min_prior = np.min(Cls_np, axis=0)
        max_prior = np.max(Cls_np, axis=0)

        ax.fill_between(range(len(min_prior)), min_prior, max_prior, alpha=0.3, label='prior')
        
        for i in range(n_cosmologies):
            ax.plot(Cls_np[i, :], label=f'cosmology {i+1}')
        
        ax.set_xlabel('$\ell$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend()
    
    plt.tight_layout()
    return fig, axs

def plot_observed_vs_predicted(true_parameter, theta_pred_TT, theta_pred_EE, theta_pred_BB, theta_pred_TE,
                                Cls_TT_prior, Cls_EE_prior, Cls_BB_prior, Cls_TE_prior, figsize=(16, 6)):
    """Crea una figura con 4 subplots mostrando espectros observados vs predichos, con bandas del prior."""
    
    # Simuladores para espectros
    simulator_TT = create_simulator("TT")
    simulator_EE = create_simulator("EE")
    simulator_BB = create_simulator("BB")
    simulator_TE = create_simulator("TE")

    # Espectros observados
    Cls_TT_obs = simulator_TT(true_parameter)
    Cls_EE_obs = simulator_EE(true_parameter)
    Cls_BB_obs = simulator_BB(true_parameter)
    Cls_TE_obs = simulator_TE(true_parameter)

    # Espectros predichos (una muestra)
    Cls_TT_pred = simulator_TT(theta_pred_TT)
    Cls_EE_pred = simulator_EE(theta_pred_EE)
    Cls_BB_pred = simulator_BB(theta_pred_BB)
    Cls_TE_pred = simulator_TE(theta_pred_TE)

    # Configuraciones de graficado
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Predicted vs observed CMB angular power spectrum', fontsize=16, y=1.02)

    plot_configs = [
        (Cls_TT_obs, Cls_TT_pred, Cls_TT_prior, '$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu K^2$]', 'TT'),
        (Cls_EE_obs, Cls_EE_pred, Cls_EE_prior, '$\ell(\ell+1)C_\ell^{EE}/2\pi$ [$\mu K^2$]', 'EE'),
        (Cls_BB_obs, Cls_BB_pred, Cls_BB_prior, '$\ell(\ell+1)C_\ell^{BB}/2\pi$ [$\mu K^2$]', 'BB'),
        (Cls_TE_obs, Cls_TE_pred, Cls_TE_prior, '$\ell(\ell+1)C_\ell^{TE}/2\pi$ [$\mu K^2$]', 'TE')
    ]

    for ax, (Cls_obs, Cls_pred, Cls_prior, ylabel, cl_type) in zip(axs.flat, plot_configs):
        Cls_obs_np = Cls_obs.numpy() if hasattr(Cls_obs, 'numpy') else Cls_obs
        Cls_pred_np = Cls_pred.numpy() if hasattr(Cls_pred, 'numpy') else Cls_pred
        Cls_prior_np = Cls_prior.numpy() if hasattr(Cls_prior, 'numpy') else Cls_prior

        min_prior = np.min(Cls_prior_np, axis=0)
        max_prior = np.max(Cls_prior_np, axis=0)

        ax.fill_between(range(len(min_prior)), min_prior, max_prior, alpha=0.3)

        ax.plot(Cls_obs_np, '--', label='observed', zorder=20, linewidth=2)
        ax.plot(Cls_pred_np, '-', label='predicted', zorder=10, linewidth=2)

        ax.set_xlabel('$\ell$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend()

    plt.tight_layout()
    return fig, axs

if __name__ == '__main__':
    import scienceplots
    plt.style.use(['science', 'bright'])
    plt.rcParams['figure.dpi'] = 300

    simulations = torch.load(os.path.join(PATHS["simulations"], "all_Cls_100000.pt"), weights_only=True)
    theta, all_Cls = simulations["theta"], simulations["x"]

    Cls_TT = Cl_XX(all_Cls, "TT")
    Cls_EE = Cl_XX(all_Cls, "EE")
    Cls_BB = Cl_XX(all_Cls, "BB")
    Cls_TE = Cl_XX(all_Cls, "TE")

    true_parameter = [0.02212, 0.1206, 1.04077, 3.04, 0.9626]

    from src.posterior import posterior_NPSE, sample_posterior

    posterior_TT = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_100000.pth"), theta, Cls_TT)
    samples_TT = sample_posterior(posterior_TT, true_parameter, type_str="TT", num_samples=1000)
    theta_pred_TT = samples_TT[500]
    print(theta_pred_TT)

    posterior_EE = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_EE_100000.pth"), theta, Cls_EE)
    samples_EE = sample_posterior(posterior_EE, true_parameter, type_str="EE", num_samples=1000)
    theta_pred_EE = samples_EE[500]
    print(theta_pred_EE)

    posterior_BB = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_BB_100000.pth"), theta, Cls_BB)
    samples_BB = sample_posterior(posterior_BB, true_parameter, type_str="BB", num_samples=1000)
    theta_pred_BB = samples_BB[500]
    print(theta_pred_BB)

    posterior_TE = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TE_100000.pth"), theta, Cls_TE)
    samples_TE = sample_posterior(posterior_TE, true_parameter, type_str="TE", num_samples=1000)
    theta_pred_TE = samples_TE[500]
    print(theta_pred_TE)

    fig, axs = plot_observed_vs_predicted(true_parameter, theta_pred_TT, theta_pred_EE, theta_pred_BB, theta_pred_TE, Cls_TT_prior=Cls_TT, Cls_EE_prior=Cls_EE, Cls_BB_prior=Cls_BB, Cls_TE_prior=Cls_TE)
    plt.savefig(os.path.join(PATHS["plot_simulations"], "cmb_aps_pred_vs_obs.pdf"), dpi=300, bbox_inches='tight')

    # fig, axs = plot_cls_subplots(Cls_TT, Cls_EE, Cls_BB, Cls_TE)
    # plt.savefig(os.path.join(PATHS["plot_simulations"], "cmb_aps_subplot.pdf"), dpi=300, bbox_inches='tight')

