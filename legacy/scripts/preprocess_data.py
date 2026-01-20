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

def main(processor, cov, name_input, name_output):
    theta, x = processor.load_simulations(name_input)
    print(theta.shape, x.shape)
    x = processor.select_components(x, TT=True)
    x = processor.select_ell(x, 30, 2478)
    print(theta.shape, x.shape)

    x = processor.add_cov_noise_batch(x, cov)
    print(x.shape)
    processor.save_simulations(theta, x, name_output)

if __name__ == "__main__":
    processor = Processor(type_str="TT+EE+BB+TE")
    dataset = Dataset()
    _, _, _, ell_high, _, lmin, lmax, _, cov_matrix = dataset.import_data()
    cov = processor.expand_cov_from_binned(cov_matrix, lmin, lmax)
    print(cov.shape)

    main(processor, cov, "test_3.pt", "test_3_cov_unbinned.pt")
    main(processor, cov, "test_4.pt", "test_4_cov_unbinned.pt")
    main(processor, cov, "test_5.pt", "test_5_cov_unbinned.pt")
    main(processor, cov, "test_6.pt", "test_6_cov_unbinned.pt")
    main(processor, cov, "test_7.pt", "test_7_cov_unbinned.pt")
    main(processor, cov, "test_8.pt", "test_8_cov_unbinned.pt")
    main(processor, cov, "test_9.pt", "test_9_cov_unbinned.pt")
    main(processor, cov, "test_10.pt", "test_10_cov_unbinned.pt")