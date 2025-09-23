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

    lmin = lmin.astype(lmin.dtype.newbyteorder('='))
    lmax = lmax.astype(lmax.dtype.newbyteorder('='))
    dell_high = dell_high.astype(dell_high.dtype.newbyteorder('='))
    cov_matrix = cov_matrix.astype(cov_matrix.dtype.newbyteorder('='))

    lmin = torch.from_numpy(lmin)
    lmax = torch.from_numpy(lmax)
    dell_high = torch.from_numpy(dell_high)
    cov_matrix = torch.from_numpy(cov_matrix)

    return lmin, lmax, cov_matrix, dell_high

def main():
    plotter = Plotter.from_config()
    processor = Processor(type_str="TT")

    true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
    simulator = processor.create_simulator()
    x_obs_sim = simulator(true_parameter1)
    x_obs_sim = x_obs_sim[30:2478]
    lmin, lmax, cov_matrix, dell_high = import_data()
    x_obs_sim = processor.bin_high_ell(x_obs_sim, lmin, lmax)
    x_obs_sim = processor.add_cov_noise(x_obs_sim, cov_matrix)
    print(x_obs_sim.shape)
    
    trainer1 = Trainer("NPSE")
    theta1, x1 = processor.load_simulations("01_TT_high_ell_binned_noise_planck_50000.pt")
    trainer1.load_posterior("01_NPSE_TT_high_ell_binned_noise_planck_50000.pth", theta1, x1)
    samples1 = trainer1.sample(x=x_obs_sim)

    trainer2 = Trainer("NPSE")
    theta2, x2 = processor.load_simulations("02_TT_high_ell_binned_noise_planck_50000.pt")
    trainer2.load_posterior("02_NPSE_TT_high_ell_binned_noise_planck_50000.pth", theta2, x2)
    samples2 = trainer2.sample(x=x_obs_sim)

    trainer3 = Trainer("NPSE")
    theta3, x3 = processor.load_simulations("03_TT_high_ell_binned_noise_planck_50000.pt")
    trainer3.load_posterior("03_NPSE_TT_high_ell_binned_noise_planck_50000.pth", theta3, x3)
    samples3 = trainer3.sample(x=x_obs_sim)


    fig = plotter.plot_confidence_contours(
        all_samples = [samples1, samples2, samples3],
        true_parameter=true_parameter1,
        sample_colors=["#550A41", "#000000", "#006FED"],
        filled=[True, False, False],
        sample_labels=["seed 1", "seed 2", "seed 3"],
    )

    plt.savefig(os.path.join(PATHS["confidence"], "seed_comparison_NPSE_TT_high_ell_binned_noise_planck_50000.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    main()


