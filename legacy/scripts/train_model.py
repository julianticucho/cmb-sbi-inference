import os
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from src.config import PATHS
from src.plotter import Plotter
from src.processor import Processor
from src.trainer import Trainer
from src.data import Dataset

def main(theta, x, output_name):
    inf = Trainer("NPSE")
    print(inf.model_params)
    _, fig, axes = inf.train(theta, x, plot=False)
    inf.save(output_name)

if __name__ == "__main__":
    processor = Processor(type_str="TT")
    # theta1, x1 = processor.load_simulations("test_1_cov_unbinned.pt")
    # theta2, x2 = processor.load_simulations("test_2_cov_unbinned.pt")
    # theta3, x3 = processor.load_simulations("test_3_cov_unbinned.pt")
    # theta4, x4 = processor.load_simulations("test_4_cov_unbinned.pt")
    # theta5, x5 = processor.load_simulations("test_5_cov_unbinned.pt")
    # theta6, x6 = processor.load_simulations("test_6_cov_unbinned.pt")
    # theta7, x7 = processor.load_simulations("test_7_cov_unbinned.pt")
    # theta8, x8 = processor.load_simulations("test_8_cov_unbinned.pt")
    # theta9, x9 = processor.load_simulations("test_9_cov_unbinned.pt")
    # theta10, x10 = processor.load_simulations("test_10_cov_unbinned.pt")
    # theta_01, x_01 = processor.concatenate_simulations(theta1, x1, theta2, x2)
    # theta_02, x_02 = processor.concatenate_simulations(theta_01, x_01, theta3, x3)
    # theta_02, x_02 = processor.concatenate_simulations(theta_02, x_02, theta4, x4)
    # theta_03, x_03 = processor.concatenate_simulations(theta_02, x_02, theta5, x5)
    # theta_03, x_03 = processor.concatenate_simulations(theta_03, x_03, theta6, x6)
    # theta_04, x_04 = processor.concatenate_simulations(theta_03, x_03, theta7, x7)
    # theta_04, x_04 = processor.concatenate_simulations(theta_04, x_04, theta8, x8)
    # theta_05, x_05 = processor.concatenate_simulations(theta_04, x_04, theta9, x9)
    # theta_05, x_05 = processor.concatenate_simulations(theta_05, x_05, theta10, x10)
    # print(theta_01.shape, x_01.shape)
    # print(theta_02.shape, x_02.shape)
    # print(theta_03.shape, x_03.shape)
    # print(theta_04.shape, x_04.shape)
    # print(theta_05.shape, x_05.shape)
    
    # main(theta_01, x_01, "NPSE_50k_cov_unbinned.pth")
    # main(theta_02, x_02, "NPSE_100k_cov_unbinned.pth")
    # main(theta_03, x_03, "NPSE_150k_cov_unbinned.pth")
    # main(theta_04, x_04, "NPSE_200k_cov_unbinned.pth")
    # main(theta_05, x_05, "NPSE_250k_cov_unbinned.pth")

    theta_old, x_old = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    main(theta_old, x_old, "NPSE_50k_old_cov_unbinned.pth")

    from scripts.plot_ppc import plot_ppc
    plot_ppc()