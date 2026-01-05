import os
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from src.config import PATHS
from src.plotter import Plotter
from src.processor import Processor
from src.trainer import Trainer
from src.data import Dataset

def unbox_data():
    dataset = Dataset()
    _, _, _, _, _, lmin, lmax, _, cov_matrix = dataset.import_data()
    return lmin, lmax, cov_matrix

def plot_ppc():
    plotter = Plotter.from_config()
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix = unbox_data()
    print(cov_matrix.shape)

    true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
    simulator_obs = processor.create_simulator()
    x_obs = simulator_obs(true_parameter1)
    x_obs = x_obs[30:2478]
    x_obs = processor.bin_high_ell(x_obs, lmin, lmax)
    x_obs = processor.add_cov_noise(x_obs, cov_matrix, seed=0)
    print(x_obs.shape)

    theta1, x1 = processor.load_simulations("test_1_cov_binned.pt")
    theta2, x2 = processor.load_simulations("test_2_cov_binned.pt")
    theta3, x3 = processor.load_simulations("test_3_cov_binned.pt")
    theta4, x4 = processor.load_simulations("test_4_cov_binned.pt")
    theta5, x5 = processor.load_simulations("test_5_cov_binned.pt")
    theta6, x6 = processor.load_simulations("test_6_cov_binned.pt")
    theta7, x7 = processor.load_simulations("test_7_cov_binned.pt")
    theta8, x8 = processor.load_simulations("test_8_cov_binned.pt")
    theta9, x9 = processor.load_simulations("test_9_cov_binned.pt")
    theta10, x10 = processor.load_simulations("test_10_cov_binned.pt")
    theta_01, x_01 = processor.concatenate_simulations(theta1, x1, theta2, x2)
    theta_02, x_02 = processor.concatenate_simulations(theta_01, x_01, theta3, x3)
    theta_02, x_02 = processor.concatenate_simulations(theta_02, x_02, theta4, x4)
    theta_03, x_03 = processor.concatenate_simulations(theta_02, x_02, theta5, x5)
    theta_03, x_03 = processor.concatenate_simulations(theta_03, x_03, theta6, x6)
    theta_04, x_04 = processor.concatenate_simulations(theta_03, x_03, theta7, x7)
    theta_04, x_04 = processor.concatenate_simulations(theta_04, x_04, theta8, x8)
    theta_05, x_05 = processor.concatenate_simulations(theta_04, x_04, theta9, x9)
    theta_05, x_05 = processor.concatenate_simulations(theta_05, x_05, theta10, x10)
    print(theta_01.shape, x_01.shape)
    print(theta_02.shape, x_02.shape)
    print(theta_03.shape, x_03.shape)
    print(theta_04.shape, x_04.shape)
    print(theta_05.shape, x_05.shape)
    
    trainer01 = Trainer("SNPE_C")
    trainer01.load_posterior("SNPE_C_50k_cov_binned.pkl", theta_01, x_01)
    samples01 = trainer01.sample(x=x_obs, num_samples=25000)

    trainer02 = Trainer("SNPE_C")
    trainer02.load_posterior("SNPE_C_100k_cov_binned.pkl", theta_02, x_02)
    samples02 = trainer02.sample(x=x_obs, num_samples=25000)

    trainer03 = Trainer("SNPE_C")
    trainer03.load_posterior("SNPE_C_150k_cov_binned.pkl", theta_03, x_03)
    samples03 = trainer03.sample(x=x_obs, num_samples=25000)

    trainer04 = Trainer("SNPE_C")
    trainer04.load_posterior("SNPE_C_200k_cov_binned.pkl", theta_04, x_04)
    samples04 = trainer04.sample(x=x_obs, num_samples=25000)

    trainer05 = Trainer("SNPE_C")
    trainer05.load_posterior("SNPE_C_250k_cov_binned.pkl", theta_05, x_05)
    samples05 = trainer05.sample(x=x_obs, num_samples=25000)

    fig = plotter.plot_confidence_contours(
        all_samples = [samples01, samples02, samples03, samples04, samples05],
        true_parameter=true_parameter1,
        sample_colors=["#E03424", "#006FED", "#550A41", "#FF0000", "#000000"], #"#000000", "#006FED", "#550A41", "#E03424"
        filled=[True, True, False, False, False],
        sample_labels=["50k", "100k", "150k", "200k", "250k"],
    )
    plt.savefig(os.path.join(PATHS["confidence"], "SNPE_C_50k_100k_150k_200k_250k_cov_binned.pdf"), bbox_inches='tight')

def plot_ppc2():
    plotter = Plotter.from_config()
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix = unbox_data()

    true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
    simulator_obs = processor.create_simulator()
    x_obs = simulator_obs(true_parameter1)
    x_obs = x_obs[30:2478]
    x_obs = processor.bin_high_ell(x_obs, lmin, lmax)
    x_obs = processor.add_cov_noise(x_obs, cov_matrix, seed=0)
    print(x_obs.shape)

    trainer = Trainer("NPSE")
    theta, x = processor.load_simulations("01_TT_high_ell_binned_noise_planck_50000.pt")
    trainer.load_posterior("01_NPSE_TT_high_ell_binned_noise_planck_50000.pth", theta, x)
    samples = trainer.sample(x=x_obs)

    fig = plotter.plot_confidence_contours(
        all_samples = [samples],
        true_parameter=true_parameter1,
        sample_colors=["#E03424"],
        filled=[True],
        sample_labels=["50k"],
    )
    plt.savefig(os.path.join(PATHS["confidence"], "01_TT_high_ell_binned_noise_planck_50000_V3.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    plot_ppc()