import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import os
from src.config import PATHS
from src.prior import get_prior


class MCMCSampler:
    def __init__(
        self,
        simulator,
        cov_matrix: torch.Tensor,
        step_size: float = 0.01,
        num_chains: int = 4,
        seed: int = 0
    ):
        self.simulator = simulator
        self.prior = get_prior(format="pyro")
        self.cov = cov_matrix
        self.inv_cov = torch.linalg.inv(self.cov)
        self.step_size = step_size
        self.num_chains = num_chains
        self.seed = seed
        self.dim = self.prior.sample((1,)).shape[1]

    def _model(self, x_obs: torch.Tensor):
        theta = pyro.sample("theta", self.prior)
        x_sim = self.simulator(theta)
        delta = x_obs - x_sim

        if delta.dim() == 1:
            delta = delta.unsqueeze(0)

        log_likelihood = -0.5 * (delta @ self.inv_cov @ delta.T).squeeze()
        pyro.factor("likelihood", log_likelihood)

    def sample(
        self,
        x_obs: torch.Tensor,
        num_samples: int = 10_000,
        burn_in: int = 1_000
    ) -> torch.Tensor:
        pyro.set_rng_seed(self.seed)

        nuts_kernel = NUTS(self._model, step_size=self.step_size, adapt_step_size=True)

        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=burn_in,
            num_chains=self.num_chains,
            disable_progbar=False,
        )

        mcmc.run(x_obs)
        posterior_samples = mcmc.get_samples(group_by_chain=True)
        print(mcmc.summary())

        samples = posterior_samples["theta"]
        return samples

    def save_samples(self, samples: torch.Tensor, filename: str):
        torch.save(samples, os.path.join(PATHS["chains"], filename))

    def load_samples(self, filename: str) -> torch.Tensor:
        filepath = os.path.join(PATHS["chains"], filename)
        return torch.load(filepath)
