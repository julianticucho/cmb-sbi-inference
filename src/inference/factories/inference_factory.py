import torch
from typing import Dict
from src.simulation import PriorFactory
from sbi.inference import SNPE_C, NPSE, FMPE, NLE, MNLE
from sbi.neural_nets import (
    posterior_nn,
    posterior_score_nn,
    likelihood_nn,
    flowmatching_nn
)


class InferenceFactory:

    @staticmethod
    def get_available_configurations() -> Dict[str, Dict[str, str]]:
        return {
            'snpe_c_default': InferenceFactory.create_snpe_c_default,
            'npse_default': InferenceFactory.create_npse_default,
            'fmpe_default': InferenceFactory.create_fmpe_default,
            'nle_default': InferenceFactory.create_nle_default,
            'mnle_default': InferenceFactory.create_mnle_default,
        }

    @staticmethod
    def get_inference(model_type: str, prior_type: str):
        models = InferenceFactory.get_available_configurations()
        if model_type not in models:
            raise ValueError(f"Model {model_type} not found, available models: {list(models.keys())}")
        prior = PriorFactory.get_prior(prior_type).to_sbi()
        return models[model_type](prior)

    @staticmethod
    def create_snpe_c_default(prior):
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_npse_default(prior):
        nn = posterior_score_nn(
            sde_type="ve",
            score_net_type="mlp",
            z_score_theta="independent",
            z_score_x="independent",
            t_embedding_dim=16,
            hidden_features=50,
        )        
        return NPSE(prior=prior, score_estimator=nn)

    @staticmethod
    def create_fmpe_default(prior):
        nn = flowmatching_nn(
            model="mlp",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=64,
            num_layers=5,
            num_blocks=5,
            num_frequencies=3,
        )
        return FMPE(prior=prior, density_estimator=nn)

    @staticmethod
    def create_nle_default(prior):
        nn = likelihood_nn(
            model="maf",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return NLE(prior=prior, density_estimator=nn)

    @staticmethod
    def create_mnle_default(prior):
        nn = likelihood_nn(
            model="maf",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return MNLE(prior=prior, density_estimator=nn)


        
        

