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

def main():
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
    x_obs = torch.tensor(x_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    print(x_obs.shape)

    # trainer0 = Trainer("NPSE")
    # theta_1, x_1 = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    # theta_2, x_2 = processor.load_simulations("02_TT_high_ell_unbinned_noise_planck_50000.pt")
    # theta0, x0 = processor.concatenate_simulations(theta_1, x_1, theta_2, x_2)
    # print(theta0.shape, x0.shape)
    # trainer0.load_posterior("NPSE_01+02_TT_high_ell_unbinned_noise_planck_100000.pth", theta0, x0)
    # samples0 = trainer0.sample(x=x_obs)
    
    # trainer1 = Trainer("NPSE")
    # theta1, x1 = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    # print(theta1.shape, x1.shape)
    # trainer1.load_posterior("01_TT_high_ell_unbinned_noise_planck_50000.pth", theta1, x1)
    # samples1 = trainer1.sample(x=x_obs)

    trainer2 = Trainer("NPSE", embedding_type="CNN")
    theta2, x2 = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    print(theta2.shape, x2.shape)
    trainer2.load_posterior("NPSE+CNN_01_TT_high_ell_unbinned_noise_planck_50000.pth", theta2, x2)
    samples2 = trainer2.sample(x=x_obs)

    # from src.mcmc import MCMCSampler
    # sampler = MCMCSampler(create_simulator(), cov_matrix, step_size=0.002)
    # samples1 = sampler.load_samples("mcmc_samples_test.pt")
    # print(samples1.shape)
    # print(samples1.shape)
    # samples1 = sampler.convert_to_flat_format(samples1)
    # print(samples1.shape)

    fig = plotter.plot_confidence_contours(
        all_samples = [samples2],
        true_parameter=true_parameter1,
        sample_colors=["#E03424"], #"#000000", "#006FED", "#550A41", "#E03424"
        filled=[True],
        sample_labels=["NPSE+CNN"],
    )

    plt.savefig(os.path.join(PATHS["confidence"], "NPSE+CNN_01_TT_high_ell_unbinned_noise_planck_50000.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    main()


