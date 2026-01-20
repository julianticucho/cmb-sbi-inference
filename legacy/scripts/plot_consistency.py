import os
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from src.config import PATHS
from src.plotter import Plotter
from src.processor import Processor
from src.trainer import Trainer

def import_data():
    path = os.path.join(PATHS["planck"],"COM_PowerSpect_CMB_R1.10.fits")
    hdul = fits.open(path)
    high_data = hdul[2].data
    cov_matrix = hdul[3].data
    
    dell_high = high_data["D_ELL"]
    lmin = high_data["LMIN"]
    lmax = high_data["LMAX"]
    err = high_data["ERR"]
    ell_high = high_data["ELL"]

    lmin = lmin.astype(lmin.dtype.newbyteorder('='))
    lmax = lmax.astype(lmax.dtype.newbyteorder('='))
    dell_high = dell_high.astype(dell_high.dtype.newbyteorder('='))
    cov_matrix = cov_matrix.astype(cov_matrix.dtype.newbyteorder('='))
    err = err.astype(err.dtype.newbyteorder('='))
    ell_high = ell_high.astype(ell_high.dtype.newbyteorder('='))


    lmin = torch.from_numpy(lmin)
    lmax = torch.from_numpy(lmax)
    dell_high = torch.from_numpy(dell_high)
    cov_matrix = torch.from_numpy(cov_matrix)
    err = torch.from_numpy(err)
    ell_high = torch.from_numpy(ell_high)

    return lmin, lmax, cov_matrix, err, ell_high

def create_simulator():
    lmin, lmax, cov_matrix, err, ell_high = import_data()
    def simulator(theta):
        processor = Processor(type_str="TT")
        simulator = processor.create_simulator()
        x = simulator(theta)
        x = x[30:2478]
        x = processor.bin_high_ell(x, lmin, lmax)
        return x
    return simulator

def main():
    plotter = Plotter.from_config()
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix, err, ell_high = import_data()

    true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
    simulator = create_simulator()
    x_obs_sim = simulator(true_parameter1)
    x_obs_sim = processor.add_cov_noise(x_obs_sim, cov_matrix)
    print(x_obs_sim.shape)
    
    trainer1 = Trainer("NPSE")
    theta1, x1 = processor.load_simulations("01_TT_high_ell_binned_noise_planck_50000.pt")
    trainer1.load_posterior("01_NPSE_TT_high_ell_binned_noise_planck_50000.pth", theta1, x1)
    samples1 = trainer1.sample(x=x_obs_sim)

    fig = plotter.plot_consistency_check(samples1, ell_high, x_obs_sim, err, simulator)

    plt.savefig(os.path.join(PATHS["consistency"], "consistency_NPSE_TT_high_ell_binned_noise_planck_50000.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    main()