import os
import numpy as np
import seaborn as sns
from sbi.analysis.plot import sbc_rank_plot, plot_tarp
from sbi.diagnostics import run_sbc, run_tarp
from getdist import plots, MCSamples
from srcOOP.processor import Processor
from srcOOP.generator import Generator
from srcOOP.trainer import Trainer  
from srcOOP.config import PATHS

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300

class Plotter:
    """Class for plotting."""
    def __init__(self, param_names, param_labels=None, limits=None):
        self.param_names = param_names
        self.param_labels = param_labels or [name.replace('_', r'\_') for name in param_names]
        self.limits = limits

    def plot_confidence_contours(self, all_samples, true_parameter,
                                 sample_labels=None, sample_colors=None,
                                 filled=True, title=None):
        """Plot confidence contours for given samples."""
        if sample_colors is None:
            sample_colors = plt.cm.tab10.colors[:len(all_samples)]

        gdist_samples = []
        for i, sample in enumerate(all_samples):
            label = sample_labels[i] if sample_labels and i < len(sample_labels) else f"Run {i+1}"
            gdist = MCSamples(
                samples=np.array(sample, copy=True),
                names=self.param_names,
                labels=self.param_labels,
                label=label
            )
            gdist.fine_bins = gdist.fine_bins_2D = 1500
            gdist_samples.append(gdist)

        g = plots.get_subplot_plotter()
        g.settings.scaling_factor = 0.1
        g.settings.solid_colors = sample_colors

        limits_dict = {self.param_names[i]: self.limits[i] for i in range(len(self.param_names))} if self.limits else None
        markers_dict = {self.param_names[i]: true_parameter[i] for i in range(len(self.param_names))}

        g.triangle_plot(
            gdist_samples,
            params=self.param_names,
            filled=filled,
            param_limits=limits_dict,
            markers=markers_dict,
            legend_labels=sample_labels,
            contour_colors=sample_colors,
            legend_loc="upper right"
        )
        if title:
            plt.suptitle(title)
        return g

    def correlation_matrix(self, samples):
        """Plot correlation matrix for given samples."""
        samples_np = samples.numpy() if isinstance(samples, np.ndarray) == False else samples
        corr_matrix = np.corrcoef(samples_np.T)

        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            square=True,
            xticklabels=self.param_names,
            yticklabels=self.param_names,
            vmin=-1, vmax=1
        )
        return fig
    
    def plot_sbc(self, thetas, xs, posterior, num_posterior_samples=1000):
        """Plot SBC rank plot."""
        ranks, dap_samples = run_sbc(
            thetas=thetas,
            xs=xs, 
            posterior=posterior, 
            num_posterior_samples=num_posterior_samples,
            num_workers=11,
            use_sample_batched=False,
        )
        fig, axes = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            num_bins=10,
            plot_type="cdf",
        )
        return fig, axes
    
    def plot_tarp(self, thetas, xs, posterior, num_posterior_samples=1000):
        """Plot TARP diagnostic."""
        ecp, alpha = run_tarp(
            thetas,
            xs,
            posterior, 
            references=None,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False
        )
        fig, axes = plot_tarp(
            ecp,
            alpha,
        )
        return fig, axes

if __name__ == "__main__":
    limits = [
        [0.02212-0.00022*(1), 0.02212+0.00022*(1)],    
        [0.1206-0.0021*(1), 0.1206+0.0021*(1)],  
        [1.04077-0.00047*(1), 1.04077+0.00047*(1)],      
        [3.04-0.016*(1), 3.04+0.016*(1)],    
        [0.9626-0.0057*(1), 0.9626+0.0057*(1)],
        # [0.0522-0.008*(3/4), 0.0522+0.008*(3/4)]  
    ]
    param_names = ['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns']
    param_labels = [r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$']

    true_parameter1 = [0.02212, 0.1206, 1.04077, 3.04, 0.9626]
    true_parameter2 = [0.02205, 0.1224, 1.04035, 3.028, 0.9589]
    true_parameter3 = [0.02218, 0.1198, 1.04052, 3.052, 0.9672]

    # -------------------------------------------------------------------------------------------------------------

    processor = Processor(type_str="TT")
    simulator = processor.create_simulator(noise=True, binning=False)
    cl_obs = simulator(true_parameter1)
    
    trainer1 = Trainer("NPSE")
    theta1, x1 = processor.load_simulations("Cls_TT_noise_reduced_prior_50000_01.pt")
    trainer1.load_posterior("NPSE_TT_noise_reduced_prior_50000_01.pth", theta1, x1)
    samples1 = trainer1.sample(type_str="TT", noise=True, binning=False, num_samples=100000, x=cl_obs)

    trainer2 = Trainer("NPSE")
    theta2, x2 = processor.load_simulations("Cls_TT_noise_reduced_prior_50000_02.pt")
    trainer2.load_posterior("NPSE_TT_noise_reduced_prior_50000_02.pth", theta2, x2)
    samples2 = trainer2.sample(type_str="TT", noise=True, binning=False, num_samples=100000, x=cl_obs)

    plotter = Plotter(param_names, param_labels, limits)
    fig = plotter.plot_confidence_contours([samples1, samples2], 
                                           true_parameter1, 
                                           sample_labels=["Seed 1", "Seed 2"], 
                                           sample_colors=["#0A5524", "#000000"],
                                           filled=[True, False],
                                           )

    plt.savefig(os.path.join(PATHS["confidence"], "NPSE_TT_low_noise_seed_comparison_50000.pdf"), bbox_inches='tight')
    plt.close('all')
    del fig

    #-------------------------------------------------------------------------------------------------------------

    gen = Generator(type_str="TT", num_workers=11, seed=1)
    processor = Processor(type_str="TT")
    trainer = Trainer("NPSE")
    plotter = Plotter(param_names, param_labels, limits)

    theta, x = gen.generate_cosmologies(num_simulations=200)
    thetas, xs = processor.generate_noise_multiple(theta, x)

    theta_train, x_train = processor.load_simulations("all_Cls_reduced_prior_50000.pt")
    x_train = processor.select_components(x_train, TT=True)

    posterior = trainer.load_posterior("NPSE_TT_reduced_prior_50000_03.pth", theta_train, x_train)
    fig, axes = plotter.plot_sbc(thetas, xs, posterior, num_posterior_samples=1000)

    plt.savefig(os.path.join(PATHS["calibration"], "sbc_NPSE_TT_reduced_prior_50000.pdf"), bbox_inches='tight')
    plt.close('all')
    del fig

    