from src.processor import Processor
from src.mcmc import MCMCSampler
from src.simulator_wraper import simulator, unbox_data

def main():
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix = unbox_data()

    true_parameter1 = [0.022068, 0.12029, 1.04122, 3.098, 0.9624]
    simulator_obs = processor.create_simulator()
    x_obs = simulator_obs(true_parameter1)
    x_obs = x_obs[30:2478]
    x_obs = processor.bin_high_ell(x_obs, lmin, lmax)
    x_obs = processor.add_cov_noise(x_obs, cov_matrix, seed=0)
    print(x_obs.shape)

    mcmc_sampler = MCMCSampler(simulator, cov_matrix, step_size=0.002)
    samples = mcmc_sampler.sample(x_obs, num_samples=15, burn_in=1)
    print(samples.shape)
    mcmc_sampler.save_samples(samples, "mcmc_samples_test.pt")

if __name__ == "__main__":
    main()