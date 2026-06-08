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
        alphas_hpd = []
        pbar = tqdm(range(n), desc="HPD diagnostic", unit="pair")
        for i in pbar:
            theta_true = thetas[i]
            x = xs[i]
            post_samples = _sample_posterior(model, x, num_posterior_samples)

            # HPD exacto usando log_prob (método Lemos et al. 2023, Algorithm 1)
            # La región HPD está definida por: {θ : p̂(θ|x) >= p̂(θ_true|x)}
            log_prob_true = model.log_prob(theta_true.unsqueeze(0), x=x)
            log_prob_samples = model.log_prob(post_samples, x=x)
            alpha_hat = (log_prob_samples >= log_prob_true).float().mean().item()
            alphas_hpd.append(alpha_hat)

        alphas_hpd = np.asarray(alphas_hpd)
        a_sorted, ecp = _pp_from_alphas(alphas_hpd)
        ax.plot(a_sorted, ecp, lw=2, label="HPD")

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

def plot_data_ppc(
    posterior_samples: torch.Tensor,
    simulator: Any,
    pipeline: Any,
    observation: torch.Tensor,
    x_axis_values: Optional[np.ndarray] = None,
    num_samples_for_plot: int = 100,
    confidence_intervals: List[float] = [0.68, 0.95],
    title: str = "Posterior Predictive Check (Data Space)",
    xlabel: str = "Bin / Index",
    ylabel: str = "Data",
    log_scale: bool = False,
    device: str = "cpu"
) -> plt.Figure:
    posterior_samples = posterior_samples.to(device)
    observation = observation.to(device)
    
    if len(posterior_samples) > num_samples_for_plot:
        indices = torch.randperm(len(posterior_samples))[:num_samples_for_plot]
        samples_to_run = posterior_samples[indices]
    else:
        samples_to_run = posterior_samples
        
    simulated_data = []
    
    # Run simulations
    print(f"Generating {len(samples_to_run)} simulations for PPC...")
    for theta in tqdm(samples_to_run, desc="Simulating PPC"):
        # 1. Simulate clean data
        # Note: Simulator might expect shape (N_params) or (1, N_params)
        try:
            x_clean = simulator.simulate(theta)
        except Exception:
             # Try batched if single fails or vice versa, but simulator.simulate usually takes single param tensor 
             # based on previous file read (power_spectrum_simulator.py: simulate takes 1D tensor)
            x_clean = simulator.simulate(theta)

        # 2. Process through pipeline (add noise, binning, etc.)
        # Pipeline .run() typically takes clean x and optional seed
        # We don't fix seed here to capture noise uncertainty unless we want to isolate parameter uncertainty
        # Usually for PPC we want predictive distribution P(x|theta) which includes noise P(x|theta, clean_x)
        x_proc = pipeline.run(x_clean)
        
        simulated_data.append(x_proc.cpu().numpy())
    
    simulated_data = np.array(simulated_data) # Shape: (N_sims, Data_Dim)
    observation_np = observation.cpu().numpy()
    
    if x_axis_values is None:
        x_axis_values = np.arange(len(observation_np))
        
    mu = np.mean(simulated_data, axis=0)
    std = np.std(simulated_data, axis=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax_main = axes[0]
    
    # Plot Confidence Intervals
    # Sort data for quantiles is safer for non-gaussian distributions
    for ci in sorted(confidence_intervals, reverse=True):
        alpha = (1 - ci) / 2
        lower = np.quantile(simulated_data, alpha, axis=0)
        upper = np.quantile(simulated_data, 1 - alpha, axis=0)
        ax_main.fill_between(x_axis_values, lower, upper, alpha=0.3, label=f'{int(ci*100)}% CI')
        
    ax_main.plot(x_axis_values, mu, 'b--', label='Mean Simulated', alpha=0.8)
    ax_main.plot(x_axis_values, observation_np, 'k-', label='Observation', linewidth=1.5)
    
    if log_scale:
        ax_main.set_yscale('log')
        
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    ax_res = axes[1]
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    
    residuals = (observation_np - mu) / std_safe
    
    ax_res.plot(x_axis_values, residuals, 'k.', markersize=4)
    ax_res.axhline(0, color='gray', linestyle='--')
    ax_res.axhline(2, color='r', linestyle=':', alpha=0.5)
    ax_res.axhline(-2, color='r', linestyle=':', alpha=0.5)
    ax_res.set_ylabel(r"$(x_{obs} - \mu_{sim}) / \sigma_{sim}$")
    ax_res.set_xlabel(xlabel)
    ax_res.grid(True, alpha=0.3)
    # Limit y-axis if residuals are huge
    # ax_res.set_ylim(-5, 5) 
    
    plt.tight_layout()
    return fig

def plot_loss_history(
    loss_history: List[float],
    validation_loss: Optional[List[float]] = None,
    log_scale: bool = True,
    title: str = "Training Loss"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, label='Training Loss')
    if validation_loss is not None:
        ax.plot(validation_loss, label='Validation Loss')
    
    if log_scale:
        ax.set_yscale('log')
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_hpd_marginal(
    model: Any,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    num_posterior_samples: int = 1000,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    param_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, dict]:
    """Plot HPD coverage for each parameter marginally (1D HPD intervals).

    For each parameter, calculates the Highest Posterior Density (HPD) credible
    intervals using the marginal posterior distribution and checks coverage.

    Args:
        model: Neural posterior model with .log_prob() and .sample() methods.
        thetas: True parameter values, shape (N, D).
        xs: Observations, shape (N, ...).
        num_posterior_samples: Number of posterior samples per observation.
        seed: Random seed for reproducibility.
        device: Computation device.
        param_labels: Labels for each parameter dimension.

    Returns:
        Tuple of (matplotlib Figure, dict with coverage results per parameter).
    """
    if thetas.ndim != 2:
        raise ValueError(f"Expected `thetas` to be 2D (N, D), got shape {tuple(thetas.shape)}")
    if xs.ndim < 1:
        raise ValueError(f"Expected `xs` to have at least 1 dimension, got shape {tuple(xs.shape)}")
    if thetas.shape[0] != xs.shape[0]:
        raise ValueError(f"Mismatched N between `thetas` and `xs`: {thetas.shape[0]} vs {xs.shape[0]}")

    if device is None:
        device = thetas.device

    thetas = thetas.to(device)
    xs = xs.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_samples, n_params = thetas.shape

    # Default percentiles (same approach as plot_hpd_tarp_diagnostics)
    percentiles = torch.linspace(1, 99, 50, device=device)
    cred_levels = percentiles / 100.0

    def _sample_posterior(posterior: Any, x: torch.Tensor, n: int) -> torch.Tensor:
        try:
            return posterior.sample((n,), x=x, show_progress_bars=False)
        except TypeError:
            return posterior.set_default_x(x).sample((n,), show_progress_bars=False)

    # For each parameter, store coverage results
    coverage_results = {f"param_{d}": torch.zeros(len(percentiles), device=device) for d in range(n_params)}

    pbar = tqdm(range(n_samples), desc="HPD marginal diagnostic", unit="sample")
    for i in pbar:
        theta_true = thetas[i]
        x = xs[i]
        post_samples = _sample_posterior(model, x, num_posterior_samples)

        # Calculate HPD marginal for each parameter (optimized with batch)
        for d in range(n_params):
            # Get marginal samples for this parameter
            samples_d = post_samples[:, d]

            # Calculate HPD credible interval using log_prob approach
            # For marginal HPD, we evaluate log_prob at points along this parameter
            # while keeping other parameters fixed at their posterior mean
            param_mean = post_samples.mean(dim=0)

            # Create grid of points for this parameter
            n_grid = 200
            param_min = samples_d.min()
            param_max = samples_d.max()
            grid_d = torch.linspace(param_min, param_max, n_grid, device=device)

            # Build batch of theta points: each row varies only parameter d
            theta_batch = param_mean.unsqueeze(0).repeat(n_grid, 1)
            theta_batch[:, d] = grid_d

            # Evaluate log_prob for all grid points in one batch call
            log_probs = model.log_prob(theta_batch, x=x)

            # Find log_prob at true value
            log_prob_true = model.log_prob(theta_true.unsqueeze(0), x=x).item()

            # For each credibility level, check if true value is in HPD region
            for idx, cred in enumerate(cred_levels):
                # Find threshold: the (1-cred) quantile of log_probs
                threshold = torch.quantile(log_probs, 1 - cred)

                # True value is in HPD if its log_prob >= threshold
                if log_prob_true >= threshold:
                    coverage_results[f"param_{d}"][idx] += 1.0

    # Normalize by number of samples
    for key in coverage_results:
        coverage_results[key] = (coverage_results[key] / n_samples).cpu().numpy()

    # Create plot
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 3 * n_params), constrained_layout=True)
    if n_params == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        label = param_labels[d] if param_labels and d < len(param_labels) else f"Param {d+1}"
        ax.plot(percentiles.cpu().numpy(), coverage_results[f"param_{d}"], 'o-', markersize=4, label=label)
        ax.plot([0, 100], [0, 1], '--', color='gray', alpha=0.5)
        ax.set_xlabel('Credible level (%)')
        ax.set_ylabel('Empirical coverage')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    return fig, coverage_results


def plot_regression_results(
    model: Any,
    test_dataloader: "torch.utils.data.DataLoader",
    param_labels: Optional[List[str]] = None,
    limits: Optional[List[Tuple[float, float]]] = None,
    device: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_true: List[torch.Tensor] = []
    all_pred: List[torch.Tensor] = []

    with torch.no_grad():
        for x_batch, theta_batch in test_dataloader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu()
            all_true.append(theta_batch.cpu())
            all_pred.append(pred)

    y_true = torch.cat(all_true, dim=0).numpy()  
    y_pred = torch.cat(all_pred, dim=0).numpy()  

    n_params = y_true.shape[1]

    if param_labels is None:
        param_labels = [f"θ_{i + 1}" for i in range(n_params)]

    n_cols = min(n_params, 3)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten() if n_params > 1 else [axes]

    for i in range(n_params):
        ax = axes_flat[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]

        # R² coefficient
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        ax.scatter(yt, yp, s=6, alpha=0.35, rasterized=True, label="Test samples")

        # Axis limits: use provided limits or fall back to data range
        if limits is not None and i < len(limits):
            lo, hi = limits[i]
        else:
            lo = min(yt.min(), yp.min())
            hi = max(yt.max(), yp.max())
        margin = (hi - lo) * 0.05

        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "--", color="gray", linewidth=1, label="Identity")

        ax.set_xlabel(f"True {param_labels[i]}")
        ax.set_ylabel(f"Predicted {param_labels[i]}")
        ax.set_title(f"{param_labels[i]}  ($R^2 = {r2:.4f}$)")
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")

    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    return fig
