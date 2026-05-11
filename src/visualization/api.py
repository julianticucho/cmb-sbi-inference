import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Any
from .diagnostic import plot_ppc, plot_hpd, plot_data_ppc, plot_hpd_tarp_diagnostics
from ..simulation.factories import SimulatorFactory, PipelineFactory
from ..inference.api import load_posterior, load_prior
from ..core import storage

def plot_and_save_ppc(
    samples: List[torch.Tensor],
    true_parameter: List[float],
    param_names: List[str],
    param_labels: Optional[List[str]] = None,
    sample_labels: Optional[List[str]] = None,
    sample_colors: Optional[List[str]] = None,
    filled: bool = True,
    title: Optional[str] = None,
    limits: Optional[List[tuple]] = None,
    output_name: Optional[str] = None
):
    # Plot ppc using plot_ppc function
    fig = plot_ppc(
        all_samples=samples,
        true_parameter=true_parameter,
        param_names=param_names,
        param_labels=param_labels,
        sample_labels=sample_labels,
        sample_colors=sample_colors,
        filled=filled,
        title=title,
        limits=limits
    )

    # Save figure if output_name is provided 
    # Return figure
    if output_name:
        storage.save_figure(fig, output_name, category="confidence")
        print(f"Saved figure to {output_name}")
    return fig

def plot_and_save_hpd(
    model_filename: str,
    simulator_type: str,
    pipeline_type: str,
    param_labels: Optional[List[str]] = None,
    num_posterior_samples: int = 5000,
    num_true_samples: int = 100,
    output_name: Optional[str] = None
):
    # Load posterior and prior from model filename
    # Create simulator and pipeline functions from factories
    # Define simulator function as lambda
    # Plot HPD using plot_hpd function
    posterior = load_posterior(model_filename)
    prior = load_prior(model_filename)
    simulator = SimulatorFactory.get_simulator(simulator_type)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)
    simulator_func = lambda theta: pipeline.simulate_example(theta, simulator)
    fig, results = plot_hpd(
        posterior=posterior,
        simulator=simulator_func,
        prior=prior,
        num_posterior_samples=num_posterior_samples,
        num_true_samples=num_true_samples,
        param_labels=param_labels
    )

    # Save figure if output_name is provided
    # Return figure and results
    if output_name:
        storage.save_figure(fig, output_name, category="hpd")
    return fig, results

def plot_and_save_data_ppc(
    posterior_samples: torch.Tensor,
    simulator_type: str,
    pipeline_type: str,
    observation: torch.Tensor,
    num_samples_for_plot: int = 100,
    log_scale: bool = False,
    confidence_intervals: List[float] = [0.68, 0.95],
    title: str = "Posterior Predictive Check (Data Space)",
    xlabel: str = "Bin / Index",
    ylabel: str = "Data",
    device: str = "cpu",
    output_filename: Optional[str] = None,
):
    # Create simulator and pipeline functions from factories
    # Plot data PPC using plot_data_ppc function
    simulator = SimulatorFactory.get_simulator(simulator_type)
    pipeline = PipelineFactory.get_pipeline(pipeline_type)
    fig = plot_data_ppc(
        posterior_samples=posterior_samples,
        simulator=simulator,
        pipeline=pipeline,
        observation=observation,
        num_samples_for_plot=num_samples_for_plot,
        log_scale=log_scale,
        confidence_intervals=confidence_intervals,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        device=device,
    )

    # Save figure if output_filename is provided
    # Return figure
    if output_filename:
        storage.save_figure(fig, output_filename, category="data_ppc")
    return fig

def plot_and_save_diagnostics(
    model_filename: str,
    simulation_files: List[str],
    run_hpd: bool = True,
    run_tarp: bool = True,
    num_posterior_samples: int = 1000,
    num_references: int = 1,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_name: Optional[str] = None,
):
    # Load posterior and simulations from files
    # Plot diagnostics using plot_hpd_tarp_diagnostics function
    model = load_posterior(model_filename)
    thetas, xs = storage.load_multiple_simulations(simulation_files)
    fig = plot_hpd_tarp_diagnostics(
        model=model, 
        thetas=thetas, 
        xs=xs, 
        run_hpd=run_hpd, 
        run_tarp=run_tarp, 
        num_posterior_samples=num_posterior_samples, 
        num_references=num_references, 
        seed=seed, 
        device=device
    )
    
    # Save figure if output_name is provided
    # Return figure
    if output_name:
        storage.save_figure(fig, output_name, category="diagnostics")
    return fig
