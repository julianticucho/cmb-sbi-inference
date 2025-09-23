import torch
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from src.config import PATHS
from src.processor import Processor
from src.trainer import Trainer
from src.plotter import Plotter

def import_data():
    path = os.path.join(PATHS["planck"],"COM_PowerSpect_CMB_R1.10.fits")
    hdul = fits.open(path)
    high_data = hdul[2].data
    cov_matrix = hdul[3].data
    
    dell_high = high_data["D_ELL"]
    lmin = high_data["LMIN"]
    lmax = high_data["LMAX"]

    lmin = lmin.astype(lmin.dtype.newbyteorder('='))
    lmax = lmax.astype(lmax.dtype.newbyteorder('='))
    cov_matrix = cov_matrix.astype(cov_matrix.dtype.newbyteorder('='))

    lmin = torch.from_numpy(lmin)
    lmax = torch.from_numpy(lmax)
    cov_matrix = torch.from_numpy(cov_matrix)

    return lmin, lmax, cov_matrix

def create_simulator():
    lmin, lmax, cov_matrix = import_data()
    def simulator(theta):
        processor = Processor(type_str="TT")
        simulator = processor.create_simulator()
        x = simulator(theta)
        x = x[30:2478]
        x = processor.bin_high_ell(x, lmin, lmax)
        x = processor.add_cov_noise(x, cov_matrix)
        return x
    return simulator

def main():
    processor = Processor(type_str="TT")
    trainer = Trainer("NPSE")
    plotter = Plotter.from_config()

    theta, x = processor.load_simulations("01_TT_high_ell_binned_noise_planck_50000.pt")
    trainer.load_posterior("01_NPSE_TT_high_ell_binned_noise_planck_50000.pth", theta, x)
    posterior = trainer.posterior  # <- distribución entrenada

    fig, coverage = plotter.plot_posterior_calibration(
        posterior=posterior,
        simulator=create_simulator(),  
        num_posterior_samples=2000,
        num_true_samples=1000,         
        device="cpu"
    )

    plt.savefig(os.path.join(PATHS["calibration"], "01_NPSE_TT_high_ell_binned_noise_planck_50000.pdf"), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()