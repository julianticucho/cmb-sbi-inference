import torch
import numpy as np
import seaborn as sns
import os
from src.posterior import posterior_SNPE_C, posterior_NPSE, sampler_mcmc, sample_posterior
from src.config import PATHS
from src.simulator import Cl_XX
from getdist import plots, MCSamples

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

def plot_confidence_contours_nsamples(all_samples, true_parameter, limits, param_names, 
                                      param_labels=None, sample_labels=None, title=None,
                                      sample_colors=None, upper_samples=None, upper_styles=None, filled=True):

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

    gdist_samples_upper = []
    if upper_samples is not None:
        for i, sample in enumerate(upper_samples):
            sample = np.array(sample, copy=True)
            label = sample_labels[i] if sample_labels and i < len(sample_labels) else f'Muestra {i+1}'
            gdist = MCSamples(samples=sample, 
                            names=param_names, 
                            labels=param_labels,
                            label=label,)
            gdist.fine_bins = 1500
            gdist.fine_bins_2D = 1500
            gdist_samples_upper.append(gdist)
 
    g = plots.get_subplot_plotter()
    g.settings.scaling_factor = 0.1
    g.settings.solid_colors = sample_colors

    limits_dict = {param_names[i]: limits[i] for i in range(len(param_names))} if limits else None
    markers_dict = {param_names[i]: true_parameter[i] for i in range(len(param_names))}

    if upper_samples is not None:
        g.triangle_plot(gdist_samples, 
                        params=param_names,
                        filled=filled,
                        param_limits=limits_dict,
                        markers=markers_dict,
                        legend_labels=sample_labels,
                        contour_colors=sample_colors,
                        upper_roots=gdist_samples_upper,
                        upper_kwargs=upper_styles or {})  
    else:
        g.triangle_plot(gdist_samples, 
                        params=param_names,
                        filled=filled,
                        param_limits=limits_dict,
                        markers=markers_dict,
                        legend_labels=sample_labels,
                        contour_colors=sample_colors,
                        legend_loc='upper right')

    return g

if __name__ == "__main__":
    limits = [
        [0.02212-0.00022*(3/4), 0.02212+0.00022*(3/4)],    
        [0.1206-0.0021*(3/4), 0.1206+0.0021*(3/4)],  
        [1.04077-0.00047*(3/4), 1.04077+0.00047*(3/4)],      
        [3.04-0.016*(3/4), 3.04+0.016*(3/4)],    
        [0.9626-0.0057*(3/4), 0.9626+0.0057*(3/4)],
        # [0.0522-0.008*(3/4), 0.0522+0.008*(3/4)]  
    ]
    param_names = ['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns']
    param_labels = [r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$']

    true_parameter1 = [0.02212, 0.1206, 1.04077, 3.04, 0.9626]
    true_parameter2 = [0.02205, 0.1224, 1.04035, 3.028, 0.9589]
    true_parameter3 = [0.02218, 0.1198, 1.04052, 3.052, 0.9672]

    # simulations1 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_noise_25000.pt"), weights_only=True)
    # simulations2 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_repeat4_noise_25000.pt"), weights_only=True)
    # simulations3 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_repeat10_noise_25000.pt"), weights_only=True)
    # simulations4 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_repeat20_noise_25000.pt"), weights_only=True)

    # theta, x_TT = simulations1["theta"], Cl_XX(simulations1["x"], "TT")
    # posterior_TT = posterior_NPSE(os.path.join(PATHS["models"], "TT+noise_NPSE_25000.pth"), theta, x_TT)
    # samples_TT = sample_posterior(posterior_TT, true_parameter3, type_str="TT+noise")
    
    # theta_repeat4, x_TT_repeat4 = simulations2["theta"], simulations2["x"]
    # posterior_TT_repeat4 = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_repeat4_noise_25000.pth"), theta_repeat4, x_TT_repeat4)
    # samples_TT_repeat4 = sample_posterior(posterior_TT_repeat4, true_parameter3, type_str="TT+noise")

    # theta_repeat10, x_TT_repeat10 = simulations3["theta"], Cl_XX(simulations3["x"], "TT")
    # posterior_TT_repeat10 = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_repeat10_noise_25000.pth"), theta_repeat10, x_TT_repeat10)
    # samples_TT_repeat10 = sample_posterior(posterior_TT_repeat10, true_parameter3, type_str="TT+noise")

    # theta_repeat20, x_TT_repeat20 = simulations4["theta"], Cl_XX(simulations4["x"], "TT")
    # posterior_TT_repeat20 = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_repeat20_noise_25000.pth"), theta_repeat20, x_TT_repeat20)
    # samples_TT_repeat20 = sample_posterior(posterior_TT_repeat20, true_parameter3, type_str="TT+noise")

    # fig = plot_confidence_contours_nsamples([samples_TT, samples_TT_repeat4, samples_TT_repeat10, samples_TT_repeat20],
    #                                          true_parameter=true_parameter3, limits=limits, 
    #                                          param_names=param_names, param_labels=param_labels,
    #                                          sample_labels=['TT','TT_repeat4','TT_repeat10', 'TT_repeat20'], 
    #                                          title='Posterior Predictive Check',
    #                                          sample_colors=['#006FED','#009966',"#000866","#000000"],
    #                                          filled=[True, True, True, False],
    #                                         #  upper_samples=[samples_2, samples_1, samples_TT],
    #                                         #  upper_styles={"contour_colors": ["gray","#000000",'#006FED'], "contour_ls": ["-", "-", "-"], "show_1d": [False, False, False], "filled": [True, False, True]}
    # )
                                                   
    # plt.savefig(os.path.join(PATHS["confidence"], "03_repeat_noise_comparison_25000.pdf"), bbox_inches='tight')
    # plt.close('all')
    # del fig

    # simulations1 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_noise_100000.pt"), weights_only=True)
    # simulations2 = torch.load(os.path.join(PATHS["simulations"], "Cls_TT_repeat5_noise_100000.pt"), weights_only=True)

    # theta, x_TT = simulations1["theta"], Cl_XX(simulations1["x"], "TT")
    # posterior_TT = posterior_NPSE(os.path.join(PATHS["models"], "TT+noise_NPSE_100000.pth"), theta, x_TT)
    # samples_TT = sample_posterior(posterior_TT, true_parameter3, type_str="TT+noise")
    
    # theta_repeat5, x_TT_repeat5 = simulations2["theta"], simulations2["x"]
    # posterior_TT_repeat5 = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_repeat5_noise_100000.pth"), theta_repeat5, x_TT_repeat5)
    # samples_TT_repeat5 = sample_posterior(posterior_TT_repeat5, true_parameter3, type_str="TT+noise")

    # fig = plot_confidence_contours_nsamples([samples_TT, samples_TT_repeat5],
    #                                          true_parameter=true_parameter3, limits=limits, 
    #                                          param_names=param_names, param_labels=param_labels,
    #                                          sample_labels=['TT','TT_repeat5'], 
    #                                          title='Posterior Predictive Check',
    #                                          sample_colors=['#006FED','#009966'],
    #                                          filled=[True, True],
    #                                         #  upper_samples=[samples_2, samples_1, samples_TT],
    #                                         #  upper_styles={"contour_colors": ["gray","#000000",'#006FED'], "contour_ls": ["-", "-", "-"], "show_1d": [False, False, False], "filled": [True, False, True]}
    # )
                                                   
    # plt.savefig(os.path.join(PATHS["confidence"], "03_repeat_noise_comparison_100000.pdf"), bbox_inches='tight')
    # plt.close('all')
    # del fig

    simulations1 = torch.load(os.path.join(PATHS["simulations"], "all_Cls_tau_50000_reduced_prior.pt"), weights_only=True)

    theta, x_TT = simulations1["theta"], Cl_XX(simulations1["x"], "TT")
    posterior_TT = posterior_NPSE(os.path.join(PATHS["models"], "NPSE_TT_reduced_prior_50000.pth"), theta, x_TT)
    samples_TT = sample_posterior(posterior_TT, true_parameter1, type_str="TT")

    fig = plot_confidence_contours_nsamples([samples_TT],
                                             true_parameter=true_parameter1, limits=limits, 
                                             param_names=param_names, param_labels=param_labels,
                                             sample_labels=['TT'], 
                                             title='Posterior Predictive Check',
                                             sample_colors=['#006FED'],
                                             filled=[True],
    )
                                                   
    plt.savefig(os.path.join(PATHS["confidence"], "01_NPSE_TT_reduced_prior_50000.pdf"), bbox_inches='tight')
    plt.close('all')
    del fig
