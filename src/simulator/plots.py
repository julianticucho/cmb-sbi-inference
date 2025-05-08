import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from src.simulator.config import PARAM_RANGES, PATHS

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

central_params = {k: (v[0] + v[1])/2 for k, v in PARAM_RANGES.items()}
n_values = 4 
params_order = ['ombh2', 'omch2', 'theta_MC_100', 'tau', 'ln_10_10_As', 'ns']
param_latex = {
    'ombh2': r'$\Omega_b h^2$',
    'omch2': r'$\Omega_c h^2$',
    'theta_MC_100': r'$100\theta_{\rm MC}$',
    'tau': r'$\tau$',
    'ln_10_10_As': r'$\ln(10^{10}A_s)$',
    'ns': r'$n_s$'
}

def plot_varying_param(simulator, output_dir):
    for param_name in params_order:
        plt.figure(figsize=(12, 8))
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

if __name__ == '__main__':
    from src.simulator.simulator import compute_spectrum, create_simulator
    plot_varying_param(compute_spectrum, PATHS['base_spectra'])
    simulator = create_simulator()
    plot_varying_param(simulator, PATHS["noisy_spectra"] )

print("Gráficos guardados")

