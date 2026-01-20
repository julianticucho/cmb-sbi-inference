#!/usr/bin/env python3
"""
Run CMB SBI experiments from configuration files.

This script provides a command-line interface for running
experiments defined in YAML configuration files.
"""
import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sbi_inference import (
    create_simulator,
    load_config,
    SimulationLoader,
    SimulationManager
)
from sbi_inference.data import CMBSimulator


def run_experiment(config_path: str) -> None:
    """
    Run experiment from configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to experiment configuration file
    """
    # Load configuration
    config = load_config(f"experiments/{config_path}")
    
    print(f"Running experiment: {config.get('name', 'Unnamed')}")
    print(f"Model: {config['model']['name']}")
    print(f"Components: {config['model']['components']}")
    
    # Create simulator
    simulator_wrapper = create_simulator(config['model']['name'])
    
    # Load simulations
    sim_loader = SimulationLoader()
    theta, x = sim_loader.load_simulations(config['data']['simulation_file'])
    
    # Process data if needed
    if config.get('processing', {}).get('noise', False):
        # Add noise processing here
        pass
    
    # Generate samples (placeholder for actual SBI logic)
    print(f"Loaded {theta.shape[0]} simulations")
    print(f"Parameter shape: {theta.shape}")
    print(f"Spectrum shape: {x.shape}")
    
    # Save results
    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Placeholder for actual results saving
    print(f"Results would be saved to: {results_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CMB SBI experiments"
    )
    parser.add_argument(
        "config",
        help="Experiment configuration file (without .yaml extension)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment configurations"
    )
    
    args = parser.parse_args()
    
    if args.list:
        configs_dir = Path("configs/experiments")
        if configs_dir.exists():
            configs = list(configs_dir.glob("*.yaml"))
            print("Available experiments:")
            for config in sorted(configs):
                print(f"  - {config.stem}")
        else:
            print("No experiments directory found")
        return
    
    try:
        run_experiment(args.config)
        print("Experiment completed successfully!")
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
