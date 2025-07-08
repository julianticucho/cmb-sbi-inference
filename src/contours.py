import torch
import numpy as np
import seaborn as sns
import os
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.posterior import posterior_SNPE_C, posterior_NPSE, sampler_mcmc, sample_posterior
from src.config import PATHS
from src.simulator import Cl_XX
from getdist import plots, MCSamples

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

def plot_confidence_contours(samples, true_parameter, limits, param_names):
    """ Grafica las posteriores de los parámetros y las regiones de confianza. """
    samples_np = np.array(samples)
    true_params_np = np.array(true_parameter)
    limits = np.array(limits)
    n_params = samples_np.shape[1]

    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    def plot_1d_dist(ax, data, true_val, limits):
        sns.kdeplot(data, ax=ax, color='black')
        ax.axvline(true_val, color='r', linestyle='--', linewidth=1)
        ax.set_xlim(limits)
        ax.set_yticklabels([])
        ax.set_yticks([])
    
    def plot_2d_contour(ax, x, y, true_x, true_y, x_lim, y_lim):
        kde = gaussian_kde(np.vstack([x, y]))

        xx, yy = np.mgrid[x_lim[0]:x_lim[1]:100j, y_lim[0]:y_lim[1]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = np.reshape(kde(positions).T, xx.shape)
        
        sorted_zz = np.sort(zz.ravel())
        cdf = np.cumsum(sorted_zz) / np.sum(sorted_zz)
        level_95 = sorted_zz[np.argmin(np.abs(cdf - 0.05))]  # 95% CI 
        level_68 = sorted_zz[np.argmin(np.abs(cdf - 0.32))]  # 68% CI 
        
        contour_levels = sorted([level_95, level_68])
        colors = ['#4b9cd3', '#a2d0fa']  
    
        ax.contourf(xx, yy, zz, levels=[contour_levels[0], contour_levels[1]], 
                   colors=colors[1], alpha=0.3)  # 95% CI
        ax.contourf(xx, yy, zz, levels=[contour_levels[1], zz.max()], 
                   colors=colors[0], alpha=0.5)  # 68% CI
        ax.contour(xx, yy, zz, levels=contour_levels, colors='#1f77b4', linewidths=0.5)
        
        ax.scatter(true_x, true_y, color='r', marker='x', s=50, label='True value')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:  
                plot_1d_dist(ax, samples_np[:, i], true_params_np[i], limits[i])
            elif i > j: 
                plot_2d_contour(ax, samples_np[:, j], samples_np[:, i], true_params_np[j], true_params_np[i], limits[j], limits[i])
            else:  
                ax.axis('off')
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i])
            if i != n_params - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
    
    legend_elements = [
        Patch(facecolor='#4b9cd3', alpha=0.5, label='68% CI'),
        Patch(facecolor='#a2d0fa', alpha=0.3, label='95% CI'),
        Line2D([0], [0], marker='x', color='r', label='True value', 
              linestyle='None', markersize=8)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.suptitle('Posterior Distribution with Confidence Contours', y=1.02)
    plt.tight_layout()
    return fig, axes

def plot_confidence_contours_2samples(samples1, samples2, true_parameter, limits, param_names, 
                                     labels=['Sample 1', 'Sample 2'], colors=['#1f77b4', '#ff7f0e']):
    """
    Grafica las posteriores de dos conjuntos de muestras en el mismo gráfico con regiones de confianza.
    """
    samples1_np = np.array(samples1)
    samples2_np = np.array(samples2)
    true_params_np = np.array(true_parameter)
    limits = np.array(limits)
    n_params = samples1_np.shape[1]
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    dash_pattern = (5, 3)  
    dark_gray = "#444444" 
    line_width = 0.5     
    line_alpha = 0.9   
    
    def plot_1d_dist(ax, data1, data2, true_val, limits):
        sns.kdeplot(data1, ax=ax, color=colors[0], label=labels[0])
        sns.kdeplot(data2, ax=ax, color=colors[1], label=labels[1])
        ax.axvline(true_val, color=dark_gray, linestyle='--', linewidth=line_width, alpha=line_alpha, dashes=dash_pattern)
        ax.set_xlim(limits)
        ax.set_yticklabels([])
        ax.set_yticks([])
    
    def plot_2d_contour(ax, x1, y1, x2, y2, true_x, true_y, x_lim, y_lim):
        
        kde1 = gaussian_kde(np.vstack([x1, y1]))
        xx, yy = np.mgrid[x_lim[0]:x_lim[1]:100j, y_lim[0]:y_lim[1]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz1 = np.reshape(kde1(positions).T, xx.shape)
        
        kde2 = gaussian_kde(np.vstack([x2, y2]))
        zz2 = np.reshape(kde2(positions).T, xx.shape)
        
        sorted_zz1 = np.sort(zz1.ravel())
        cdf1 = np.cumsum(sorted_zz1) / np.sum(sorted_zz1)
        level_95_1 = sorted_zz1[np.argmin(np.abs(cdf1 - 0.05))]
        level_68_1 = sorted_zz1[np.argmin(np.abs(cdf1 - 0.32))]
        contour_levels1 = sorted([level_95_1, level_68_1])
        
        sorted_zz2 = np.sort(zz2.ravel())
        cdf2 = np.cumsum(sorted_zz2) / np.sum(sorted_zz2)
        level_95_2 = sorted_zz2[np.argmin(np.abs(cdf2 - 0.05))]
        level_68_2 = sorted_zz2[np.argmin(np.abs(cdf2 - 0.32))]
        contour_levels2 = sorted([level_95_2, level_68_2])
        
        ax.contour(xx, yy, zz1, levels=contour_levels1, colors=colors[0], linewidths=0.5, alpha=0.7)
        ax.contour(xx, yy, zz2, levels=contour_levels2, colors=colors[1], linewidths=0.5, alpha=0.7)
        
        ax.contourf(xx, yy, zz1, levels=[contour_levels1[0], contour_levels1[1]], 
                   colors=[colors[0]], alpha=0.1)  
        ax.contourf(xx, yy, zz1, levels=[contour_levels1[1], zz1.max()], 
                   colors=[colors[0]], alpha=0.2)  
        
        ax.contourf(xx, yy, zz2, levels=[contour_levels2[0], contour_levels2[1]], 
                   colors=[colors[1]], alpha=0.1)  
        ax.contourf(xx, yy, zz2, levels=[contour_levels2[1], zz2.max()], 
                   colors=[colors[1]], alpha=0.2)  
        
        ax.axhline(true_y, color=dark_gray, linestyle='--', linewidth=line_width, alpha=line_alpha, dashes=dash_pattern)
        ax.axvline(true_x, color=dark_gray, linestyle='--', linewidth=line_width, alpha=line_alpha, dashes=dash_pattern)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:  
                plot_1d_dist(ax, samples1_np[:, i], samples2_np[:, i], 
                            true_params_np[i], limits[i])
            elif i > j: 
                plot_2d_contour(ax, 
                               samples1_np[:, j], samples1_np[:, i],
                               samples2_np[:, j], samples2_np[:, i],
                               true_params_np[j], true_params_np[i], 
                               limits[j], limits[i])
            else:  
                ax.axis('off')
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i])
            if i != n_params - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
    
    legend_line = Line2D([0], [0], color=dark_gray, linestyle='--', 
                        linewidth=line_width, alpha=line_alpha, dashes=dash_pattern, label="True value")
    
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label=labels[0]),
        Line2D([0], [0], color=colors[1], lw=2, label=labels[1]),
        legend_line  
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.suptitle('Posterior Distribution Comparison', y=1.02)
    plt.tight_layout()
    return fig, axes

def plot_confidence_contours_nsamples(all_samples, true_parameter, limits, param_names, 
                                      param_labels=None, sample_labels=None, title=None,
                                      sample_colors=None):

    if param_labels is None:
        param_labels = [name.replace('_', r'\_') for name in param_names]
    if sample_colors is None:
        sample_colors = plt.cm.tab10.colors[:len(all_samples)]

    gdist_samples = []
    for i, sample in enumerate(all_samples):
        sample = np.array(sample, copy=True)
        label = sample_labels[i] if sample_labels and i < len(sample_labels) else f'Muestra {i+1}'
        gdist = MCSamples(samples=sample, 
                          names=param_names, 
                          labels=param_labels,
                          label=label,)
        gdist.fine_bins = 1500
        gdist.fine_bins_2D = 1500
        gdist_samples.append(gdist)

    g = plots.get_subplot_plotter()
    g.settings.scaling_factor = 0.1
    g.settings.solid_colors = sample_colors

    limits_dict = {param_names[i]: limits[i] for i in range(len(param_names))} if limits else None
    markers_dict = {param_names[i]: true_parameter[i] for i in range(len(param_names))}

    g.triangle_plot(gdist_samples, 
                    params=param_names,
                    filled=True,
                    param_limits=limits_dict,
                    markers=markers_dict,
                    legend_labels=sample_labels,
                    legend_loc='upper right',
                    contour_colors=sample_colors
                    )
    if title is not None:
        g.fig.suptitle(title, y=1.03)

    return g

if __name__ == "__main__":
    limits = [
        [0.02212-0.00022*2, 0.02212+0.00022*2],    
        [0.1206-0.0021*2, 0.1206+0.0021*2],  
        [1.04077-0.00047*2, 1.04077+0.00047*2],      
        [3.04-0.016*2, 3.04+0.016*2],    
        [0.9626-0.0057*2, 0.9626+0.0057*2],
        # [0.0522-0.008, 0.0522+0.008]  
    ]
    param_names = ['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns']
    param_labels = [r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$']

    true_parameter1 = [0.02212, 0.1206, 1.04077, 3.04, 0.9626]
    true_parameter2 = [0.02205, 0.1224, 1.04035, 3.028, 0.9589]
    true_parameter3 = [0.02218, 0.1198, 1.04052, 3.052, 0.9672]
    
    simulations_1 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_noise_binned_100000.pt"), weights_only=True)
    theta_1, x_1 = simulations_1["theta"], simulations_1["cl"]
    print(theta_1.shape, x_1.shape)

    posterior_1 = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_noise_binned_100000.pth"), theta_1, x_1)
    samples_1 = sample_posterior(posterior_1, true_parameter1, type_str="TT+noise+binned")

    simulations_2 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_noise_100000.pt"), weights_only=True)
    theta_2, x_2 = simulations_2["theta"], simulations_2["x"]
    print(theta_2.shape, x_2.shape)

    posterior_2 = posterior_NPSE(os.path.join(PATHS["models"], "TT+noise_NPSE_100000.pth"), theta_2, x_2)
    samples_2 = sample_posterior(posterior_2, true_parameter1, type_str="TT+noise")

    all_samples = [samples_1, samples_2]
    fig =plot_confidence_contours_nsamples(all_samples, true_parameter1, limits, param_names, param_labels, sample_labels=['TT+noise+binned', 'TT+noise'], sample_colors=["#CE2323", "#A2D233"], title=r"Comparison of inference with and without binning in noise APS")
    plt.savefig(os.path.join(PATHS["confidence"], "inference_with_noise.pdf"), bbox_inches='tight')
    plt.close('all')
    del fig

    # theta, x_TT = simulations["theta"], Cl_XX(simulations["x"], "TT")
    # posterior_TT = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_100000.pth"), theta, x_TT)
    # samples_TT = sample_posterior(posterior_TT, true_parameter1, type_str="TT")

    # theta, x_EE = simulations["theta"], Cl_XX(simulations["x"], "EE")
    # posterior_EE = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_EE_100000.pth"), theta, x_EE)
    # samples_EE = sample_posterior(posterior_EE, true_parameter1, type_str="EE")

    # theta, x_BB = simulations["theta"], Cl_XX(simulations["x"], "BB")
    # posterior_BB = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_BB_100000.pth"), theta, x_BB)
    # samples_BB = sample_posterior(posterior_BB, true_parameter1, type_str="BB")

    # theta, x_TE = simulations["theta"], Cl_XX(simulations["x"], "TE")
    # posterior_TE = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TE_100000.pth"), theta, x_TE)
    # samples_TE = sample_posterior(posterior_TE, true_parameter1, type_str="TE")

    # fig = plot_confidence_contours_nsamples([samples_BB, samples_TE, samples_EE, samples_TT],
    #                                          true_parameter=true_parameter1, limits=limits, 
    #                                          param_names=param_names, param_labels=param_labels,
    #                                          sample_labels=['BB', 'TE', 'EE', 'TT'], 
    #                                          title='Comparison of posterior depending of type of data',
    #                                          sample_colors=['#336600''#E03424', '#009966',"#000866",'#006FED'])
                                                   
    # plt.savefig(os.path.join(PATHS["confidence"], "data_comparison.pdf"), bbox_inches='tight')
    # plt.close('all')
    # del fig
