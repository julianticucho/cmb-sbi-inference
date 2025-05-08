import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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