import torch
import sbi
import os
import matplotlib.pyplot as plt
from srcOOP.prior import get_prior
from srcOOP.processor import Processor
from srcOOP.config import PATHS
from typing import Optional, Dict, Tuple

class Trainer:
    def __init__(self, method: str = "NPSE", embedding_type: Optional[str] = None, model_params: Optional[Dict] = None):
        """Class for handling density estimator training, loading and sampling."""
        self.method = method
        self.embedding_type = embedding_type 
        self.model_params = model_params or {}
        self.inference = self._create_inference()
        self.density_estimator = None
        self.posterior = None

    def _create_inference(self):
        """Create a sbi inference object based on the selected method."""
        prior = get_prior()
        net = self._create_network()

        if self.method == "SNPE_C":
            return sbi.inference.SNPE_C(prior=prior, density_estimator=net)
        elif self.method == "NPSE":
            return sbi.inference.NPSE(prior=prior, score_estimator=net)
        elif self.method == "FMPE":
            return sbi.inference.FMPE(prior=prior, density_estimator=net)
        elif self.method == "NLE":
            return sbi.inference.NLE(prior=prior, density_estimator=net)
        elif self.method == "MNLE":
            return sbi.inference.MNLE(prior=prior, density_estimator=net)
        else:
            raise ValueError(f"Unrecognized inference method: {self.method}, use 'SNPE_C', 'NPSE', 'FMPE', 'NLE' or 'MNLE'")

    def _create_network(self):
        """Create a network based on the selected method."""
        embeding_net = self._create_embedding(self.embedding_type)

        if self.method in ["SNPE_C", "FMPE"]:
            return sbi.neural_nets.posterior_nn(model=self.model_params.get("model", "nsf"), embedding_net=embeding_net)
        elif self.method == "NPSE":
            return sbi.neural_nets.posterior_score_nn(sde_type=self.model_params.get("sde_type", "ve"), embedding_net=embeding_net)
        elif self.method in ["NLE", "MNLE"]:
            return sbi.neural_nets.likelihood_nn(model=self.model_params.get("model", "nsf"), embedding_net=embeding_net)
        else:
            raise ValueError(f"Unrecognized inference method: {self.method}, use 'SNPE_C', 'NPSE', 'FMPE', 'NLE' or 'MNLE'")
        
    def _create_embedding(self, embedding_type):
        """Create an embedding network based on the selected type."""
        if embedding_type is None:
            return torch.nn.Identity()
        elif embedding_type == "FCE":
            return sbi.neural_nets.embedding_nets.FCEmbedding(input_dim=2551)
        elif embedding_type == "CNN":
            return sbi.neural_nets.embedding_nets.CNNEmbedding(input_shape=(2551,))
        else:
            raise ValueError(f"Embedding net not recognized: {embedding_type}, use 'identity', 'FCE' or 'CNN'")

    def train(self, theta: torch.Tensor, x: torch.Tensor, plot: bool = True) -> Tuple[torch.nn.Module, Optional[plt.Figure], Optional[plt.Axes]]:
        """Train the density estimator on the given data."""
        self.inference.append_simulations(theta, x)
        self.density_estimator = self.inference.train()
        fig, axes = (None, None)
        if plot:
            fig, axes = sbi.analysis.plot_summary(self.inference, tags=["training_loss", "validation_loss"], figsize=(10,2))
        return self.density_estimator, fig, axes

    def save(self, filename: str):
        """Save the trained density estimator"""
        if self.density_estimator is None:
            raise RuntimeError("Density estimator is not trained.")
        torch.save(self.density_estimator.state_dict(), os.path.join(PATHS["models"], filename))

    def load_posterior(self, filename: str, theta: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None):
        """Load the trained density estimator and build the posterior"""
        path = os.path.join(PATHS["models"], filename)
        if theta is not None and x is not None:
            self.inference.append_simulations(theta, x)
        self.density_estimator = self.inference.train(max_num_epochs=0)
        self.density_estimator.load_state_dict(torch.load(path, weights_only=True))
        return self.build_posterior()

    def build_posterior(self):
        """Create the posterior from the trained density estimator"""
        if self.density_estimator is None:
            raise RuntimeError("Density estimator is not trained.")
        self.posterior = self.inference.build_posterior(self.density_estimator)
        return self.posterior

    def sample(self, type_str: str = "TT+EE+BB+TE", noise: bool = False, binning: bool = False, num_samples: int = 24000, true_parameter: Optional[torch.Tensor]=None, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the posterior"""
        if self.posterior is None:
            raise RuntimeError("Posterior is not built. Call build_posterior() first.")
        processor = Processor(type_str=type_str)
        if x is None:
            simulator = processor.create_simulator(noise=noise, binning=binning)
            x = simulator(true_parameter)
        return self.posterior.set_default_x(x).sample((num_samples,))
    

if __name__ == "__main__":
    processor = Processor(type_str="TT+EE+BB+TE", K=1)
    theta, x = processor.load_simulations("all_Cls_reduced_prior_50000.pt")
    tt = processor.select_components(x, TT=True)
    theta, tt_noise = processor.generate_noise_multiple(theta, tt)
    processor.save_simulations(theta, tt_noise, "Cls_TT_noise_reduced_prior_50000_01.pt")
    
    inf = Trainer("NPSE")
    inf.train(theta, tt_noise, plot=False)
    inf.save("NPSE_TT_noise_reduced_prior_50000_01.pth")

    processor = Processor(type_str="TT+EE+BB+TE", K=1)
    theta, x = processor.load_simulations("all_Cls_reduced_prior_50000.pt")
    tt = processor.select_components(x, TT=True)
    theta, tt_noise = processor.generate_noise_multiple(theta, tt)
    processor.save_simulations(theta, tt_noise, "Cls_TT_noise_reduced_prior_50000_02.pt")
    
    inf = Trainer("NPSE")
    inf.train(theta, tt_noise, plot=False)
    inf.save("NPSE_TT_noise_reduced_prior_50000_02.pth")
    
    

        