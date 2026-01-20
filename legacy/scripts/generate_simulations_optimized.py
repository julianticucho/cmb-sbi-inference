"""
Generate CMB simulations using the refactored architecture with simulate_for_sbi.

This script demonstrates the optimized way to generate simulations using
sbi.simulate_for_sbi for parallelization and reproducibility.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sbi_inference import (
    create_simulator,
    get_prior,
    SimulationManager
)
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.inference import simulate_for_sbi

def generate_and_save_simulations_optimized(
    num_simulations: int = 10,
    components: str = "TT",
    config_name: str = "default",
    output_filename: str = "test_simulations_optimized.pt",
    num_workers: int = 4,
    seed: int = 42
):
    """
    Generate and save CMB simulations using simulate_for_sbi.
    
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
    num_workers : int
        Number of parallel workers for simulation
    seed : int
        Random seed for reproducibility
    """
    print(f"🚀 Generating {num_simulations} CMB simulations (optimized)...")
    print(f"   Components: {components}")
    print(f"   Config: {config_name}")
    print(f"   Workers: {num_workers}")
    print(f"   Seed: {seed}")
    print(f"   Output: {output_filename}")
    
    # 1. Create prior
    print("\n1️⃣ Creating prior...")
    prior = get_prior(device='cpu')
    print(f"   ✅ Prior: {type(prior).__name__}")
    
    # 2. Create simulator
    print("\n2️⃣ Creating simulator...")
    simulator_wrapper = create_simulator(config=config_name)
    print(f"   ✅ Simulator: {type(simulator_wrapper).__name__}")
    
    # 3. Process prior and simulator for sbi
    print("\n3️⃣ Processing prior and simulator for sbi...")
    prior_processed, _, prior_returns_numpy = process_prior(prior)
    simulator_processed = process_simulator(
        simulator_wrapper.simulator_function, 
        prior_processed, 
        prior_returns_numpy
    )
    print(f"   ✅ Processing completed")
    
    # 4. Generate simulations using simulate_for_sbi
    print(f"\n4️⃣ Generating simulations with simulate_for_sbi...")
    print(f"   🚀 Using {num_workers} workers for parallel processing")
    
    theta, x = simulate_for_sbi(
        simulator_processed, 
        proposal=prior_processed, 
        num_simulations=num_simulations, 
        num_workers=num_workers, 
        seed=seed
    )
    
    print(f"   ✅ Parameters shape: {theta.shape}")
    print(f"   ✅ Spectra shape: {x.shape}")
    print(f"   📊 Parameter ranges:")
    for i, name in enumerate(['ombh2', 'omch2', 'theta_MC_100', 'ln_10_10_As', 'ns']):
        print(f"      {name}: [{theta[:, i].min():.4f}, {theta[:, i].max():.4f}]")
    
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
    print(f"   🚀 Parallel workers used: {num_workers}")
    print(f"   🎲 Seed used: {seed}")
    
    return theta, x

def compare_methods(num_simulations: int = 50):
    """
    Compare sequential vs parallel simulation methods.
    
    Parameters
    ----------
    num_simulations : int
        Number of simulations for comparison
    """
    print(f"\n🔄 Comparing methods with {num_simulations} simulations...")
    
    import time
    
    # Method 1: Sequential (current script)
    print("\n--- Sequential Method ---")
    start_time = time.time()
    theta_seq, x_seq = generate_and_save_simulations_optimized(
        num_simulations=num_simulations,
        output_filename="sequential_test.pt",
        num_workers=1,  # Sequential
        seed=42
    )
    seq_time = time.time() - start_time
    print(f"⏱️  Sequential time: {seq_time:.2f} seconds")
    
    # Method 2: Parallel (optimized)
    print("\n--- Parallel Method ---")
    start_time = time.time()
    theta_par, x_par = generate_and_save_simulations_optimized(
        num_simulations=num_simulations,
        output_filename="parallel_test.pt", 
        num_workers=4,  # Parallel
        seed=42
    )
    par_time = time.time() - start_time
    print(f"⏱️  Parallel time: {par_time:.2f} seconds")
    
    # Comparison
    print(f"\n📊 Performance Comparison:")
    speedup = seq_time / par_time
    print(f"   Sequential: {seq_time:.2f}s")
    print(f"   Parallel (4 workers): {par_time:.2f}s")
    print(f"   🚀 Speedup: {speedup:.2f}x")
    
    # Verify results are similar (due to same seed)
    diff_theta = torch.max(torch.abs(theta_seq - theta_par))
    diff_x = torch.max(torch.abs(x_seq - x_par))
    print(f"   🎲 Max difference (theta): {diff_theta:.2e}")
    print(f"   🎲 Max difference (x): {diff_x:.2e}")
    
    return seq_time, par_time

def verify_saved_simulations(filename: str):
    """Verify that simulations were saved correctly."""
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
        description="Generate CMB simulations (optimized with simulate_for_sbi)"
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
        default="test_simulations_optimized.pt",
        help="Output filename"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify saved simulations after generation"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare sequential vs parallel methods"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison
        compare_methods(args.num_simulations)
    else:
        # Generate optimized simulations
        theta, x = generate_and_save_simulations_optimized(
            num_simulations=args.num_simulations,
            components=args.components,
            config_name=args.config,
            output_filename=args.output,
            num_workers=args.num_workers,
            seed=args.seed
        )
        
        # Verify if requested
        if args.verify:
            verify_saved_simulations(args.output)
        
        print(f"\n🎉 Optimized simulation generation completed!")
        print(f"💡 Next step: Use these simulations for SBI training")
        print(f"🚀 Performance: Used {args.num_workers} workers with seed {args.seed}")

if __name__ == "__main__":
    main()
