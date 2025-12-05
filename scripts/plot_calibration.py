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

def main(theta, x, input_name_model, output_name):
    trainer = Trainer("NPSE")
    plotter = Plotter.from_config()
    print(theta.shape, x.shape)

    trainer.load_posterior(input_name_model, theta, x)
    posterior = trainer.posterior  

    fig, coverage = plotter.plot_posterior_calibration(
        posterior=posterior,
        simulator=create_simulator(),  
        num_posterior_samples=2000,
        num_true_samples=1000,         
        device="cpu"
    )

    plt.savefig(os.path.join(PATHS["calibration"], output_name), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    processor = Processor(type_str="TT")
    theta1, x1 = processor.load_simulations("test_1_cov_unbinned.pt")
    theta2, x2 = processor.load_simulations("test_2_cov_unbinned.pt")
    theta3, x3 = processor.load_simulations("test_3_cov_unbinned.pt")
    theta4, x4 = processor.load_simulations("test_4_cov_unbinned.pt")
    theta5, x5 = processor.load_simulations("test_5_cov_unbinned.pt")
    theta6, x6 = processor.load_simulations("test_6_cov_unbinned.pt")
    theta7, x7 = processor.load_simulations("test_7_cov_unbinned.pt")
    theta8, x8 = processor.load_simulations("test_8_cov_unbinned.pt")
    theta9, x9 = processor.load_simulations("test_9_cov_unbinned.pt")
    theta10, x10 = processor.load_simulations("test_10_cov_unbinned.pt")
    theta_01, x_01 = processor.concatenate_simulations(theta1, x1, theta2, x2)
    theta_02, x_02 = processor.concatenate_simulations(theta_01, x_01, theta3, x3)
    theta_02, x_02 = processor.concatenate_simulations(theta_02, x_02, theta4, x4)
    theta_03, x_03 = processor.concatenate_simulations(theta_02, x_02, theta5, x5)
    theta_03, x_03 = processor.concatenate_simulations(theta_03, x_03, theta6, x6)
    theta_04, x_04 = processor.concatenate_simulations(theta_03, x_03, theta7, x7)
    theta_04, x_04 = processor.concatenate_simulations(theta_04, x_04, theta8, x8)
    theta_05, x_05 = processor.concatenate_simulations(theta_04, x_04, theta9, x9)
    theta_05, x_05 = processor.concatenate_simulations(theta_05, x_05, theta10, x10)
    
    main(theta_05, x_05, "NPSE_250k_cov_unbinned.pth", "NPSE_250k_cov_unbinned.pdf")
    main(theta_04, x_04, "NPSE_200k_cov_unbinned.pth", "NPSE_200k_cov_unbinned.pdf")
    main(theta_03, x_03, "NPSE_150k_cov_unbinned.pth", "NPSE_150k_cov_unbinned.pdf")
    main(theta_02, x_02, "NPSE_100k_cov_unbinned.pth", "NPSE_100k_cov_unbinned.pdf")
    main(theta_01, x_01, "NPSE_50k_cov_unbinned.pth", "NPSE_50k_cov_unbinned.pdf")
