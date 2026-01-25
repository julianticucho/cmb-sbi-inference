import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Any
from sbi.analysis.plot import sbc_rank_plot, plot_tarp
from sbi.diagnostics import run_sbc, run_tarp
from getdist import plots, MCSamples
from tqdm.auto import tqdm


def plot_ppc(
    all_samples: List[torch.Tensor], 
    true_parameter: List[float],       
    param_names: List[str],
    param_labels: Optional[List[str]] = None,
    sample_labels: Optional[List[str]] = None, 
    sample_colors: Optional[List[str]] = None,
    filled: bool = True, 
    title: Optional[str] = None,
    limits: Optional[List[tuple]] = None,
    gdist_fine_bins: int = 1500,
    gdist_scaling_factor: float = 0.1,
):
    if sample_colors is None:
        sample_colors = plt.cm.tab10.colors[:len(all_samples)]
    
    if param_labels is None:
        param_labels = [name.replace('_', r'\_') for name in param_names]

    gdist_samples = []
    for i, sample in enumerate(all_samples):
        label = sample_labels[i] if sample_labels and i < len(sample_labels) else f"Run {i+1}"
        gdist = MCSamples(
            samples=np.array(sample, copy=True),
            names=param_names,
            labels=param_labels,
            label=label
        )
        gdist.fine_bins = gdist.fine_bins_2D = gdist_fine_bins
        gdist_samples.append(gdist)

    g = plots.get_subplot_plotter()
    g.settings.scaling_factor = gdist_scaling_factor
    g.settings.solid_colors = sample_colors

    limits_dict = {param_names[i]: limits[i] for i in range(len(param_names))} if limits else {}
    markers_dict = {param_names[i]: true_parameter[i] for i in range(len(param_names))}

    g.triangle_plot(
        gdist_samples,
        params=param_names,
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
    
def plot_hpd(
    posterior: torch.distributions.Distribution,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    num_posterior_samples: int = 5000,
    percentiles: Optional[torch.Tensor] = None,
    device: str = "cpu",
    param_labels: Optional[list] = None
) -> Tuple[plt.Figure, dict]:
        
        if percentiles is None:
            percentiles = torch.linspace(1, 99.9, 100, device=device)  
        
        true_parameters = thetas
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
            x = xs[i]
            posterior_samples = posterior.set_default_x(x).sample((num_posterior_samples,), show_progress_bars=False)
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
            label = param_labels[j] if param_labels and j < len(param_labels) else f"Parameter {j+1}"
            ax.plot(percentiles.cpu().numpy(), coverage_results[:, j].cpu().numpy(),
                    'o', markersize=4, linewidth=1, label=label, alpha=0.7)
        
        ax.plot([0, 100], [0, 100], '--', color='gray')
        ax.set_xlabel('Credible interval')
        ax.set_ylabel('Percentage of true values inside')
        ax.legend(loc='lower right')
        
        return fig, {p.item(): cov for p, cov in zip(percentiles, coverage_results)}
    
def plot_sbc(
    thetas: torch.Tensor, 
    xs: torch.Tensor, 
    posterior: torch.distributions.Distribution, 
    num_posterior_samples: int = 1000
) -> Tuple[plt.Figure, plt.Axes]:
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
    
def plot_tarp(
    thetas: torch.Tensor, 
    xs: torch.Tensor, 
    posterior: torch.distributions.Distribution, 
    num_posterior_samples: int = 1000
) -> Tuple[plt.Figure, plt.Axes]:
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


def plot_hpd_tarp_diagnostics(
    model: Any,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    run_hpd: bool = True,
    run_tarp: bool = True,
    num_posterior_samples: int = 1000,
    num_references: int = 1,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> plt.Figure:
    if not run_hpd and not run_tarp:
        raise ValueError("At least one of `run_hpd` or `run_tarp` must be True.")
    if thetas.ndim != 2:
        raise ValueError(f"Expected `thetas` to be 2D (N, D), got shape {tuple(thetas.shape)}")
    if xs.ndim < 1:
        raise ValueError(f"Expected `xs` to have at least 1 dimension, got shape {tuple(xs.shape)}")
    if thetas.shape[0] != xs.shape[0]:
        raise ValueError(
            f"Mismatched N between `thetas` and `xs`: {thetas.shape[0]} vs {xs.shape[0]}"
        )
    if num_posterior_samples < 2:
        raise ValueError("`num_posterior_samples` must be >= 2.")
    if num_references < 1:
        raise ValueError("`num_references` must be >= 1.")

    if device is None:
        device = thetas.device

    thetas = thetas.to(device)
    xs = xs.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _sample_posterior(posterior: Any, x: torch.Tensor, n: int) -> torch.Tensor:
        try:
            return posterior.sample((n,), x=x, show_progress_bars=False)
        except TypeError:
            return posterior.set_default_x(x).sample((n,), show_progress_bars=False)

    n = thetas.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    def _pp_from_alphas(alpha_hats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_sorted = np.sort(alpha_hats)
        ecp = np.arange(1, len(a_sorted) + 1) / len(a_sorted)
        return a_sorted, ecp

    if run_hpd:
        if num_posterior_samples < 3:
            raise ValueError("HPD diagnostic requires `num_posterior_samples >= 3`.")

        k = max(2, int(np.sqrt(num_posterior_samples)))
        k = min(k, num_posterior_samples - 1)

        alphas_hpd = []
        pbar = tqdm(range(n), desc="HPD diagnostic", unit="pair")
        for i in pbar:
            theta_true = thetas[i]
            x = xs[i]
            post_samples = _sample_posterior(model, x, num_posterior_samples)

            dmat = torch.cdist(post_samples, post_samples)
            dmat.fill_diagonal_(float("inf"))
            d_k = dmat.kthvalue(k, dim=1).values
            score_samples = -d_k

            d_true = torch.norm(post_samples - theta_true, p=2, dim=-1)
            d_true_k = d_true.kthvalue(k).values
            score_true = -d_true_k

            alpha_hat = (score_samples >= score_true).float().mean().item()
            alphas_hpd.append(alpha_hat)

        alphas_hpd = np.asarray(alphas_hpd)
        a_sorted, ecp = _pp_from_alphas(alphas_hpd)
        ax.plot(a_sorted, ecp, lw=2, label="HPD (kNN approx)")

    if run_tarp:
        alphas_tarp = []
        pbar = tqdm(range(n), desc="TARP diagnostic", unit="pair")
        for i in pbar:
            theta_true = thetas[i]
            x = xs[i]
            post_samples = _sample_posterior(model, x, num_posterior_samples)

            for _ in range(num_references):
                ridx = torch.randint(0, num_posterior_samples, (1,), device=device).item()
                ref = post_samples[ridx]
                d_true = torch.norm(theta_true - ref, p=2)
                d_samps = torch.norm(post_samples - ref, p=2, dim=-1)
                alpha_hat = (d_samps <= d_true).float().mean().item()
                alphas_tarp.append(alpha_hat)

        alphas_tarp = np.asarray(alphas_tarp)
        a_sorted, ecp = _pp_from_alphas(alphas_tarp)
        ax.plot(a_sorted, ecp, lw=2, label="TARP")

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("Nominal credibility $\\alpha$")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    return fig

def plot_hpd_legacy(
    posterior: torch.distributions.Distribution,
    simulator: callable,
    prior: torch.distributions.Distribution,
    num_posterior_samples: int = 5000,
    num_true_samples: int = 100,
    percentiles: Optional[torch.Tensor] = None,
    device: str = "cpu",
    param_labels: Optional[list] = None
) -> Tuple[plt.Figure, dict]:
        
        if percentiles is None:
            percentiles = torch.linspace(1, 99.9, 100, device=device)  
        
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
            x = simulator(theta_true)
            posterior_samples = posterior.set_default_x(x).sample((num_posterior_samples,), show_progress_bars=False)
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
            label = param_labels[j] if param_labels and j < len(param_labels) else f"Parameter {j+1}"
            ax.plot(percentiles.cpu().numpy(), coverage_results[:, j].cpu().numpy(),
                    'o', markersize=4, linewidth=1, label=label, alpha=0.7)
        
        ax.plot([0, 100], [0, 100], '--', color='gray')
        ax.set_xlabel('Credible interval')
        ax.set_ylabel('Percentage of true values inside')
        ax.legend(loc='lower right')
        
        return fig, {p.item(): cov for p, cov in zip(percentiles, coverage_results)}
