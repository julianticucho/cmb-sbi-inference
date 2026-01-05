import torch    
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from src.processor import Processor
from src.config import PATHS
from src.data import Dataset

def plot_example():
    processor = Processor()
    theta, x = processor.load_simulations("test_1.pt")
    print(theta.shape, x.shape)

    for i in range(0,10):
        plt.plot(x[i, :], label=r"realization ${i+1}$")
    plt.show()

def bin_high_ell(processor, name_input, name_output, lmin, lmax):
    theta, x = processor.load_simulations(name_input)
    print(theta.shape, x.shape)
    x = processor.select_components(x, TT=True)
    x = processor.select_ell(x, 30, 2478)
    x = processor.bin_high_ell_batch(x, lmin, lmax)
    print(theta.shape, x.shape)
    processor.save_simulations(theta, x, name_output)

def add_noise_to_binned_spectra(processor, cov, name_input, name_output):
    theta, x = processor.load_simulations(name_input)
    print(theta.shape, x.shape)
    x = processor.add_cov_noise_batch(x, cov)
    print(theta.shape, x.shape)
    processor.save_simulations(theta, x, name_output)

if __name__ == "__main__":
    processor = Processor(type_str="TT+EE+BB+TE")
    dataset = Dataset()
    _, _, _, ell_high, _, lmin, lmax, _, cov_matrix = dataset.import_data()
    print(cov_matrix.shape)

    add_noise_to_binned_spectra(processor, cov_matrix, "test_1_binned.pt", "test_1_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_2_binned.pt", "test_2_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_3_binned.pt", "test_3_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_4_binned.pt", "test_4_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_5_binned.pt", "test_5_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_6_binned.pt", "test_6_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_7_binned.pt", "test_7_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_8_binned.pt", "test_8_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_9_binned.pt", "test_9_cov_binned.pt")
    add_noise_to_binned_spectra(processor, cov_matrix, "test_10_binned.pt", "test_10_cov_binned.pt")
