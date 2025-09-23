import os
import pytest
import matplotlib.pyplot as plt
from src.processor import Processor
from src.generator import Generator
from src.plotter import Plotter
from src.trainer import Trainer
from src.config import PATHS

def test_plot_spectra():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    theta, x = gen.generate_cosmologies(num_simulations=2)
    processor = Processor(type_str="TT", K=2)
    theta_expanded, x_noisy = processor.generate_noise_multiple(theta, x)
    plt.plot(x[0, :], label="original")
    for i in range(0,2):
        plt.plot(x_noisy[i, :], label=r"realization ${i+1}$")
    plt.show()

def test_plot_binned_spectra():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    theta, x = gen.generate_cosmologies(num_simulations=2)
    processor = Processor(type_str="TT", K=1)
    theta_expanded, x_noisy = processor.generate_noise_multiple(theta, x)
    x_binned =processor.bin_spectra(x, bin_width=500)
    plt.plot(x_binned[0, :], label="binned")
    plt.show()

def test_plot_training_loss():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    theta, x = gen.generate_cosmologies(num_simulations=5)
    inf = Trainer(method="NPSE")
    density_estimator, fig, axes = inf.train(theta, x, plot=True)
    plt.show()

@pytest.mark.parametrize("noise", [True, False])
def test_plot_create_simulator_postprocessor_noise(noise):
    processor = Processor(type_str="TT", K=1)
    simulator = processor.create_simulator(noise=noise)

    theta = [0.022, 0.12, 1.041, 3.0, 0.965]
    plt.plot(simulator(theta))
    plt.show()

@pytest.mark.parametrize("binning", [500, False])
def test_plot_create_simulator_postprocessor_binning(binning):
    processor = Processor(type_str="TT", K=1)
    simulator = processor.create_simulator(binning=binning)

    theta = [0.022, 0.12, 1.041, 3.0, 0.965]
    plt.plot(simulator(theta))
    plt.show()

def test_plot_sbc():
    inf = Trainer(method="NPSE")
    processor = Processor(type_str="TT+EE+BB+TE", K=1)
    theta, x = processor.load_simulations("all_Cls_reduced_prior_50000.pt")
    tt = processor.select_components(x, TT=True)
    posterior = inf.load_posterior("NPSE_TT_reduced_prior_50000_03.pth", theta, tt)

    plotter = Plotter(
        param_names=['omega_b', 'omega_c', 'theta_MC', 'ln10As', 'ns'],
        param_labels=[r'$\omega_b$', r'$\omega_c$', r'$100\theta_{MC}$', r'$\ln(10^{10}A_s)$', r'$n_s$'],
        limits=[
                [0.02212-0.00022*(1/3), 0.02212+0.00022*(1/3)],    
                [0.1206-0.0021*(1/3), 0.1206+0.0021*(1/3)],  
                [1.04077-0.00047*(1/3), 1.04077+0.00047*(1/3)],      
                [3.04-0.016*(1/3), 3.04+0.016*(1/3)],    
                [0.9626-0.0057*(1/3), 0.9626+0.0057*(1/3)],
        ]
    )
    gen = Generator(type_str="TT", num_workers=11, seed=1)
    theta_sbc, x_sbc = gen.generate_cosmologies(num_simulations=200)
    fig, axes = plotter.plot_sbc(theta_sbc, x_sbc, posterior)
    plt.savefig(os.path.join(PATHS["calibration"], "sbc_example.pdf"), bbox_inches='tight')
    plt.close('all')
    del fig






