import torch
from typing import Dict
from src.simulation import PriorFactory
from sbi.inference import SNPE_C, NPSE, FMPE, NLE, MNLE, NRE_C
from sbi.neural_nets import (
    posterior_nn,
    likelihood_nn,
    posterior_flow_nn,
    classifier_nn,
    posterior_score_nn
)


class NeuralInferenceFactory:

    @staticmethod
    def get_available_configurations() -> Dict[str, Dict[str, str]]:
        return {
            'snpe_c_default': NeuralInferenceFactory.create_snpe_c_default,
            'npse_default': NeuralInferenceFactory.create_npse_default,
            'fmpe_default': NeuralInferenceFactory.create_fmpe_default,
            'nle_default': NeuralInferenceFactory.create_nle_default,
            'mnle_default': NeuralInferenceFactory.create_mnle_default,
            'nre_c_default': NeuralInferenceFactory.create_nre_c_default,
            'npse_regular': NeuralInferenceFactory.create_npse_regular,
            'fmpe_regular': NeuralInferenceFactory.create_fmpe_regular
        }

    @staticmethod
    def get_inference(model_type: str, prior_type: str):
        models = NeuralInferenceFactory.get_available_configurations()
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
            model="mlp",
            sde_type="ve",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=100,
            num_layers=5,
            time_emb_type="sinusoidal",
            t_embedding_dim=32,
        )        
        return NPSE(prior=prior, score_estimator=nn)

    @staticmethod
    def create_fmpe_default(prior):
        nn = posterior_flow_nn(
            model="mlp",
            sde_type="ve",
            z_score_theta=None,
            z_score_x="independent",
            hidden_features=100,
            num_layers=5,
            time_emb_type="sinusoidal",
            t_embedding_dim=32,
        )
        return FMPE(prior=prior, vf_estimator=nn)

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
            model="mnle",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return MNLE(prior=prior, density_estimator=nn)
    
    @staticmethod
    def create_nre_c_default(prior):
        nn = classifier_nn(
            model="resnet",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
        )
        return NRE_C(prior=prior, classifier=nn)
    
    @staticmethod
    def create_npse_regular(prior):  
        nn = posterior_score_nn(
            model="mlp",
            sde_type="ve",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_layers=3,
            time_emb_type="sinusoidal",
            t_embedding_dim=16,
        )     
        return NPSE(prior=prior, score_estimator=nn)
    
    def create_fmpe_regular(prior):
        nn = posterior_flow_nn(
            model="mlp",
            sde_type="ve",
            z_score_theta=None,
            z_score_x="independent",
            hidden_features=50,
            num_layers=3,
            time_emb_type="sinusoidal",
            t_embedding_dim=16,
        )
        return FMPE(prior=prior, vf_estimator=nn)
    


        
        

