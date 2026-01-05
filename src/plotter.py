import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sbi.analysis.plot import sbc_rank_plot, plot_tarp
from sbi.diagnostics import run_sbc, run_tarp
from getdist import plots, MCSamples
from src.processor import Processor
from src.generator import Generator
from src.trainer import Trainer  
from src.config import PATHS, CONFIG
from src.prior import get_prior
from typing import Optional, Tuple, Dict, Callable
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'high-vis'])
plt.rcParams['figure.dpi'] = 300

class Plotter:
    """Class for plotting."""
    def __init__(self, param_names: list, param_labels: list=None, limits: list=None):
        """
        Initialize a Plotter instance.

        Parameters
        ----------
        param_names : list
            List of parameter names.
        param_labels : list, optional
            List of parameter labels, by default None.
            If None, it is set to the parameter names with '_' replaced by '\\_'.
        limits : list, optional
            List of limits for each parameter, by default None.
        """
        self.param_names = param_names
        self.param_labels = param_labels or [name.replace('_', r'\_') for name in param_names]
        self.limits = limits

    @classmethod
    def from_config(cls):
        """Create a Plotter instance using values from the CONFIG dictionary."""
        return cls(
            param_names=CONFIG["param_names"],
            param_labels=CONFIG["param_labels"],
            limits=CONFIG["limits"],
        )

    def plot_confidence_contours(
        self, 
        all_samples: list, 
        true_parameter: list,       
        sample_labels: list=None, 
        sample_colors: list=None,
        filled: list=True, 
        title: str=None
    ):
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

    def correlation_matrix(self, samples: torch.Tensor):
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
    
    @staticmethod
    def plot_sbc(
        thetas: torch.Tensor, 
        xs: torch.Tensor, 
        posterior: torch.distributions.Distribution, 
        num_posterior_samples: int=1000
    ):
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
    
    @staticmethod
    def plot_tarp(
        thetas: torch.Tensor, xs: torch.Tensor, 
        posterior: torch.distributions.Distribution, 
        num_posterior_samples: int=1000
    ):
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
    
    def plot_consistency_check(
        self, 
        samples: torch.Tensor,
        ell: torch.Tensor, 
        cl_obs: torch.Tensor,
        cl_obs_err: torch.Tensor, 
        simulator: callable
    ):
        """Plot consistency check."""
        num_params = len(self.param_names)
        fig, axes = plt.subplots(num_params, 1, figsize=(8, 2*num_params))
        for i, ax in enumerate(axes):
            mean = torch.mean(samples, dim=0)
            std = torch.std(samples, dim=0)
            mean_param = mean.clone()
            mean_param[i] = mean[i] + std[i]
            mean_param_minus = mean.clone()
            mean_param_minus[i] = mean[i] - std[i]
            ax.errorbar(x=ell, y=cl_obs, yerr=cl_obs_err, marker='o', linestyle='', markersize=0.5, color="#006FED")
            ax.plot(simulator(mean), color='#000000', linewidth=0.1)
            ax.plot(simulator(mean_param), color='#000000', linewidth=0.1)
            ax.plot(simulator(mean_param_minus), color='#000000', linewidth=0.1)
            ax.set_title(self.param_labels[i])
        return fig
    
    @staticmethod
    def plot_CMB_map(
        map: torch.Tensor, 
        c_min: float=-400, 
        c_max: float=400, 
        x_width: float=(2**10)*(0.5)/60, 
        y_width: float=(2**10)*(0.5)/60
    ):
        """Plot CMB map."""
        map_np = map.numpy() if isinstance(map, np.ndarray) == False else map
        print("map mean:", np.mean(map_np), "map rms:", np.std(map_np))

        plt.gcf().set_size_inches(x_width, y_width)
        im = plt.imshow(map_np, interpolation='bilinear', origin='lower', cmap=cm.RdBu_r)
        im.set_clim(c_min, c_max)
        im.set_extent([0, x_width, 0, y_width])
        plt.ylabel(r"Angle $[^\circ]$")
        plt.xlabel(r"Angle $[^\circ]$")

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label="Temperature [$\mu $K]")

        fig = plt.gcf()
        return fig
    
    def plot_posterior_calibration(
        self,
        posterior: torch.distributions.Distribution,
        simulator: callable,
        num_posterior_samples: int = 5000,
        num_true_samples: int = 100,
        percentiles: torch.Tensor = None,
        device: str = "cpu"
    ):
        """Plot posterior calibration: percentage of times true parameters fall within
        posterior credible intervals. The true parameters are sampled from the prior."""
        
        if percentiles is None:
            percentiles = torch.linspace(1, 99.9, 100, device=device)  
        
        prior = get_prior(device=device)
        true_parameters = prior.sample((num_true_samples,))
        n_samples, n_params = true_parameters.shape
        
        low_percentiles = (100 - percentiles) / 2
        high_percentiles = 100 - low_percentiles
        
        all_percentiles = torch.cat([low_percentiles, high_percentiles])
        all_percentiles = torch.unique(all_percentiles)
        all_percentiles, _ = torch.sort(all_percentiles)
        
        all_percentiles_prob = all_percentiles / 100
        coverage_results = torch.zeros(len(percentiles), n_params, device=device)
        
        pbar = tqdm(total=n_samples, desc="Generating posterior samples", 
                    unit="sample", unit_scale=True)
        
        for i in range(n_samples):
            theta_true = true_parameters[i]
            x = simulator(theta_true.unsqueeze(0))
            posterior_samples = posterior.set_default_x(x).sample((num_posterior_samples,))
            percentile_values = torch.quantile(
                posterior_samples, 
                all_percentiles_prob,
                dim=0
            )
            
            percentile_dict = {p.item(): val for p, val in zip(all_percentiles, percentile_values)}
            for idx, p in enumerate(percentiles):
                low = (100 - p) / 2
                high = 100 - low
                
                lower_bounds = percentile_dict[low.item()]
                upper_bounds = percentile_dict[high.item()]
                
                within_interval = (theta_true >= lower_bounds) & (theta_true <= upper_bounds)
                coverage_results[idx] += within_interval.float()
            
            pbar.update(1)
            pbar.set_postfix({
                "current": f"{i+1}/{n_samples}",
                "samples": f"{num_posterior_samples} each"
            })
        
        pbar.close()
        coverage_results = (coverage_results / n_samples) * 100
        
        fig, ax = plt.subplots(figsize=(8, 8))
        for j in range(n_params):
            ax.plot(percentiles.cpu().numpy(), coverage_results[:, j].cpu().numpy(),
                    'o', markersize=4, linewidth=1, label=self.param_labels[j], alpha=0.7)
        
        ax.plot([0, 100], [0, 100], '--', color='gray')
        ax.set_xlabel('Credible interval')
        ax.set_ylabel('Percentage of true values inside')
        ax.legend(loc='lower right')
        
        return fig, {p.item(): cov for p, cov in zip(percentiles, coverage_results)}
            
    def plot_posterior_calibration_residuals(
        self,
        posterior: torch.distributions.Distribution,
        simulator: callable,
        num_posterior_samples: int = 5000,
        num_true_samples: int = 100,
        percentiles: torch.Tensor = None,
        device: str = "cpu"
    ):
        """
        Plot posterior calibration residuals.

        One subplot per parameter (stacked vertically).
        x-axis: Credible interval (%)
        y-axis: (observed_coverage - expected_coverage) / expected_coverage
        """
        if percentiles is None:
            percentiles = torch.linspace(1, 99.9, 100, device=device)

        prior = get_prior(device=device)
        true_parameters = prior.sample((num_true_samples,))
        n_samples, n_params = true_parameters.shape

        low_percentiles = (100 - percentiles) / 2
        high_percentiles = 100 - low_percentiles

        all_percentiles = torch.cat([low_percentiles, high_percentiles])
        all_percentiles = torch.unique(all_percentiles)
        all_percentiles, _ = torch.sort(all_percentiles)
        all_percentiles_prob = all_percentiles / 100

        coverage_results = torch.zeros(len(percentiles), n_params, device=device)

        pbar = tqdm(total=n_samples, desc="Generating posterior samples")

        for i in range(n_samples):
            theta_true = true_parameters[i]
            x = simulator(theta_true.unsqueeze(0))

            posterior_samples = posterior.set_default_x(x).sample(
                (num_posterior_samples,)
            )

            percentile_values = torch.quantile(
                posterior_samples,
                all_percentiles_prob,
                dim=0
            )

            percentile_dict = {
                p.item(): val for p, val in zip(all_percentiles, percentile_values)
            }

            for idx, p in enumerate(percentiles):
                low = (100 - p) / 2
                high = 100 - low

                lower_bounds = percentile_dict[low.item()]
                upper_bounds = percentile_dict[high.item()]

                within = (theta_true >= lower_bounds) & (theta_true <= upper_bounds)
                coverage_results[idx] += within.float()

            pbar.update(1)

        pbar.close()

        coverage_results = (coverage_results / n_samples) * 100
        expected = percentiles.unsqueeze(1)
        residuals = (coverage_results - expected) / expected
        fig, axes = plt.subplots(
            n_params, 1,
            figsize=(7, 2.5 * n_params),
            sharex=True
        )
        if n_params == 1:
            axes = [axes]
        xvals = percentiles.cpu().numpy()
        for j, ax in enumerate(axes):
            yvals = residuals[:, j].cpu().numpy()

            ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
            ax.plot(
                xvals,
                yvals,
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.8
            )

            ax.set_ylabel(
                r"$\frac{\mathrm{coverage}-\mathrm{CI}}{\mathrm{CI}}$"
            )
            ax.set_title(self.param_labels[j])

        axes[-1].set_xlabel("Credible interval")

        fig.tight_layout()

        return fig, {
            "percentiles": percentiles,
            "coverage": coverage_results,
            "residuals": residuals
        }

    


    