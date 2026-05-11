"""
Generate CMB simulations using the refactored architecture.

This script demonstrates how to create and save CMB power spectrum simulations
using the new modular structure.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# add src to Python path for imports (e.g., for local package imports)
sys.path.insert(0, str(Path(__file__).parent.parent / "src")) 

from sbi_inference import (
    create_simulator,
    get_prior,
    SimulationManager
)

def generate_and_save_simulations(
    num_simulations: int = 10,
    components: str = "TT",
    config_name: str = "default",
    output_filename: str = "test_simulations.pt"
):
    """
    Generate and save CMB simulations.
    
    Parameters
    ----------
    num_simulations : int
        Number of simulations to generate
    components : str
        CMB components to simulate
    config_name : str
        Configuration name
    output_filename : str
        Output filename
    """
    print(f"   Generating {num_simulations} CMB simulations...")
    print(f"   Components: {components}")
    print(f"   Config: {config_name}")
    print(f"   Output: {output_filename}")
    
    # 1. create prior
    print("1. Creating prior...")
    prior = get_prior(device='cpu')
    print(f" Prior: {type(prior).__name__}")
    
    # 2. create simulator
    print("2. Creating simulator...")
    simulator = create_simulator()  # using default config
    print(f" Simulator: {type(simulator).__name__}")
    
    # 3. sample parameters from prior
    print("3. Sampling parameters from prior...")
    theta = prior.sample((num_simulations,))
    print(f"   Parameters shape: {theta.shape}")
    print(f"   Parameter ranges:")
    for i, name in enumerate(['ombh2', 'omch2', 'theta_MC_100', 'ln_10_10_As', 'ns']):
        print(f"      {name}: [{theta[:, i].min():.4f}, {theta[:, i].max():.4f}]")
    
    # 4. generate spectra
    print(f"4. Generating spectra for {num_simulations} parameter sets...")
    simulator_func = simulator.simulator_function
    
    x_list = []
    for i, params in enumerate(theta):
        if i % 5 == 0:
            print(f"   Progress: {i+1}/{num_simulations}")
        spectrum = simulator_func(params)
        x_list.append(spectrum)
    
    x = torch.stack(x_list)
    print(f"   ✅ Spectra shape: {x.shape}")
    
    # 5. Save simulations
    print(f"\n5️⃣ Saving simulations to {output_filename}...")
    manager = SimulationManager()
    manager.save_simulations(theta, x, output_filename)
    print(f"   ✅ Saved successfully!")
    
    # 6. Show statistics
    print(f"\n📈 Simulation Statistics:")
    print(f"   Number of simulations: {num_simulations}")
    print(f"   Parameter dimensions: {theta.shape[1]}")
    print(f"   Spectrum length: {x.shape[1]}")
    print(f"   Mean spectrum value: {x.mean():.2e}")
    print(f"   Spectrum std: {x.std():.2e}")
    
    return theta, x

def verify_saved_simulations(filename: str):
    """
    Verify that simulations were saved correctly.
    
    Parameters
    ----------
    filename : str
        Filename to verify
    """
    print(f"\n🔍 Verifying saved simulations: {filename}")
    
    try:
        manager = SimulationManager()
        theta_loaded, x_loaded = manager.load_simulations(filename)
        
        print(f"   ✅ Loaded theta shape: {theta_loaded.shape}")
        print(f"   ✅ Loaded x shape: {x_loaded.shape}")
        print(f"   ✅ Verification successful!")
        
        return theta_loaded, x_loaded
    except Exception as e:
        print(f"   ❌ Error loading simulations: {e}")
        return None, None

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate CMB simulations"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10,
        help="Number of simulations to generate"
    )
    parser.add_argument(
        "--components",
        type=str,
        default="TT",
        help="CMB components (TT, EE, BB, TE or combinations)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_simulationss.pt",
        help="Output filename"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify saved simulations after generation"
    )
    
    args = parser.parse_args()
    
    # Generate simulations
    theta, x = generate_and_save_simulations(
        num_simulations=args.num_simulations,
        components=args.components,
        config_name=args.config,
        output_filename=args.output
    )
    
    # Verify if requested
    if args.verify:
        verify_saved_simulations(args.output)
    
    print(f"\n🎉 Simulation generation completed!")
    print(f"💡 Next step: Use these simulations for SBI training")

if __name__ == "__main__":
    main()
