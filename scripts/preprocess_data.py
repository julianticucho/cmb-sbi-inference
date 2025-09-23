import torch    
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from src.processor import Processor
from src.config import PATHS

def import_data():
    path = os.path.join(PATHS["planck"],"COM_PowerSpect_CMB_R1.10.fits")
    hdul = fits.open(path)
    high_data = hdul[2].data
    cov_matrix = hdul[3].data

    lmin = high_data["LMIN"]
    lmax = high_data["LMAX"]

    lmin = lmin.astype(lmin.dtype.newbyteorder('='))
    lmax = lmax.astype(lmax.dtype.newbyteorder('='))
    cov_matrix = cov_matrix.astype(cov_matrix.dtype.newbyteorder('='))

    lmin = torch.from_numpy(lmin)
    lmax = torch.from_numpy(lmax)
    cov_matrix = torch.from_numpy(cov_matrix)

    return lmin, lmax, cov_matrix

def plot_example():
    processor = Processor()
    theta, x = processor.load_simulations("01_TT_high_ell_binned_noise_planck_50000.pt")
    print(theta.shape, x.shape)

    for i in range(0,10):
        plt.plot(x[i, :], label=r"realization ${i+1}$")
    plt.show()


def main():
    processor = Processor(type_str="TT+EE+BB+TE")
    theta, x = processor.load_simulations("03_all_Cls_reduced_prior_50000.pt")
    x = processor.select_components(x, TT=True)
    x = processor.select_ell(x, 30, 2478)

    lmin, lmax, cov_matrix = import_data()
    x = processor.bin_high_ell_batch(x, lmin, lmax)
    processor.save_simulations(theta, x, "03_TT_high_ell_binned_planck_50000.pt")

    x = processor.add_cov_noise_batch(x, cov_matrix)
    processor.save_simulations(theta, x, "03_TT_high_ell_binned_noise_planck_50000.pt")

if __name__ == "__main__":
    main()