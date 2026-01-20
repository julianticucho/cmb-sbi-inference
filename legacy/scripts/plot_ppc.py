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

def create_simulator():
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix = unbox_data()
    cov = processor.expand_cov_from_binned(cov_matrix, lmin, lmax)
    def simulator(theta):
        simulator = processor.create_simulator()
        x = simulator(theta)
        x = x[30:2478]
        x = processor.add_cov_noise(x, cov)
        return x
    return simulator

def plot_ppc():
    plotter = Plotter.from_config()
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix = unbox_data()
    print(cov_matrix.shape)
    cov = processor.expand_cov_from_binned(cov_matrix, lmin, lmax)

    true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
    simulator_obs = processor.create_simulator()
    x_obs = simulator_obs(true_parameter1)
    x_obs = x_obs[30:2478]
    x_obs = processor.add_cov_noise(x_obs, cov, seed=0)
    print(x_obs.shape)

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
    print(theta_01.shape, x_01.shape)
    
    # trainer01 = Trainer("NPSE")
    # trainer01.load_posterior("NPSE_50k_cov_unbinned.pth", theta_01, x_01)
    # samples01 = trainer01.sample(x=x_obs)

    trainerOld = Trainer("NPSE")
    theta_old, x_old = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    trainerOld.load_posterior("01_TT_high_ell_unbinned_noise_planck_50000.pth", theta_old, x_old)
    samplesOld = trainerOld.sample(x=x_obs)

    trainerOld2 = Trainer("NPSE")
    theta_old2, x_old2 = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    trainerOld2.load_posterior("NPSE_50k_old_cov_unbinned.pth", theta_old2, x_old2)
    samplesOld2 = trainerOld2.sample(x=x_obs)

    # trainer02 = Trainer("NPSE")
    # trainer02.load_posterior("NPSE_100k_cov_unbinned.pth", theta_02, x_02)
    # samples02 = trainer02.sample(x=x_obs)

    # trainer03 = Trainer("NPSE")
    # trainer03.load_posterior("NPSE_150k_cov_unbinned.pth", theta_03, x_03)
    # samples03 = trainer03.sample(x=x_obs)

    # trainer04 = Trainer("NPSE")
    # trainer04.load_posterior("NPSE_200k_cov_unbinned.pth", theta_04, x_04)
    # samples04 = trainer04.sample(x=x_obs)

    # trainer05 = Trainer("NPSE")
    # trainer05.load_posterior("NPSE_250k_cov_unbinned.pth", theta_05, x_05)
    # samples05 = trainer05.sample(x=x_obs)

    # fig = plotter.plot_confidence_contours(
    #     all_samples = [samples01, samples02, samples03, samples04, samples05],
    #     true_parameter=true_parameter1,
    #     sample_colors=["#E03424", "#550A41", "#006FED", "#000000", "#FF0000"], #"#000000", "#006FED", "#550A41", "#E03424"
    #     filled=[True, False, True, False, True],
    #     sample_labels=["50k", "100k", "150k", "200k", "250k"],
    # )
    # plt.savefig(os.path.join(PATHS["confidence"], "NPSE_50k_100k_150k_200k_250k_cov_unbinned.pdf"), bbox_inches='tight')

    fig = plotter.plot_confidence_contours(
        all_samples = [samplesOld, samplesOld2],
        true_parameter=true_parameter1,
        sample_colors=["#E03424", "#006FED"], #"#000000", "#006FED", "#550A41", "#E03424"
        filled=[True, False],
        sample_labels=["50k old1", "50k old2"],
    )
    plt.savefig(os.path.join(PATHS["confidence"], "NPSE_50k_new_vs_oldd_cov_unbinned.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    plot_ppc()


