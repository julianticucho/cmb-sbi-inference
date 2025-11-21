import torch
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from src.config import PATHS
from src.processor import Processor
from src.trainer import Trainer
from src.plotter import Plotter
from src.data import Dataset 

def create_simulator():
    processor = Processor(type_str="TT")
    dataset = Dataset()
    _, _, _, ell_high, _, lmin, lmax, _, cov_matrix = dataset.import_data()
    cov = processor.expand_cov_from_binned(cov_matrix, lmin, lmax)
    def simulator(theta):
        simulator = processor.create_simulator()
        x = simulator(theta)
        x = x[30:2478]
        x = processor.add_cov_noise(x, cov)
        return x
    return simulator

def main():
    processor = Processor(type_str="TT")
    trainer = Trainer("NPSE")
    plotter = Plotter.from_config()

    theta_1, x_1 = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    theta_2, x_2 = processor.load_simulations("02_TT_high_ell_unbinned_noise_planck_50000.pt")
    theta, x = processor.concatenate_simulations(theta_1, x_1, theta_2, x_2)
    print(theta.shape, x.shape)
    trainer.load_posterior("NPSE_01+02_TT_high_ell_unbinned_noise_planck_100000.pth", theta, x)
    posterior = trainer.posterior  

    fig, coverage = plotter.plot_posterior_calibration(
        posterior=posterior,
        simulator=create_simulator(),  
        num_posterior_samples=2000,
        num_true_samples=1000,         
        device="cpu"
    )

    plt.savefig(os.path.join(PATHS["calibration"], "NPSE_01+02_TT_high_ell_unbinned_noise_planck_100000.pdf"), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
