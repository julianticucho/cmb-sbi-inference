import torch    
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from src.processor import Processor
from src.config import PATHS
from src.data import Dataset

def plot_example():
    processor = Processor()
    theta, x = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_100.pt")
    print(theta.shape, x.shape)

    for i in range(0,10):
        plt.plot(x[i, :], label=r"realization ${i+1}$")
    plt.show()


def main():
    processor = Processor(type_str="TT+EE+BB+TE")
    dataset = Dataset()
    _, _, _, ell_high, _, lmin, lmax, _, cov_matrix = dataset.import_data()
    cov = processor.expand_cov_from_binned(cov_matrix, lmin, lmax)
    print(cov.shape)

    theta, x = processor.load_simulations("02_all_Cls_reduced_prior_50000.pt")
    print(theta.shape, x.shape)
    x = processor.select_components(x, TT=True)
    x = processor.select_ell(x, 30, 2478)
    print(theta.shape, x.shape)

    x = processor.add_cov_noise_batch(x, cov)
    print(x.shape)
    processor.save_simulations(theta, x, "02_TT_high_ell_unbinned_noise_planck_50000.pt")

    theta, x = processor.load_simulations("03_all_Cls_reduced_prior_50000.pt")
    print(theta.shape, x.shape)
    x = processor.select_components(x, TT=True)
    x = processor.select_ell(x, 30, 2478)
    print(theta.shape, x.shape)

    x = processor.add_cov_noise_batch(x, cov)
    print(x.shape)
    processor.save_simulations(theta, x, "03_TT_high_ell_unbinned_noise_planck_50000.pt")

if __name__ == "__main__":
    main()