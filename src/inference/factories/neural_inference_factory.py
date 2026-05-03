from typing import Dict, Callable
from ...simulation import PriorFactory
from sbi.inference import SNPE_C, NPSE, FMPE, NLE, MNLE, NRE_C
from sbi.neural_nets import (
    posterior_nn,
    likelihood_nn,
    posterior_flow_nn,
    classifier_nn,
    posterior_score_nn
)
from sbi.neural_nets.embedding_nets import (
    CNNEmbedding,
    ResNetEmbedding1D,
)


class NeuralInferenceFactory:

    @staticmethod
    def get_available_configurations() -> Dict[str, Callable]:
        return {
            'snpe_c_default': NeuralInferenceFactory.create_snpe_c_default,
            'npse_default': NeuralInferenceFactory.create_npse_default,
            'fmpe_default': NeuralInferenceFactory.create_fmpe_default,
            'nle_default': NeuralInferenceFactory.create_nle_default,
            'mnle_default': NeuralInferenceFactory.create_mnle_default,
            'nre_c_default': NeuralInferenceFactory.create_nre_c_default,
            'npse_regular': NeuralInferenceFactory.create_npse_regular,
            'fmpe_regular': NeuralInferenceFactory.create_fmpe_regular,
            'snpe_c_mdn': NeuralInferenceFactory.create_snpe_c_mdn,
            'snpe_c_made': NeuralInferenceFactory.create_snpe_c_made,
            'snpe_c_maf_rqs': NeuralInferenceFactory.create_snpe_c_maf_rqs,
            'snpe_c_nsf': NeuralInferenceFactory.create_snpe_c_nsf,
            'nle_mdn': NeuralInferenceFactory.create_nle_mdn,
            'nle_made': NeuralInferenceFactory.create_nle_made,
            'nle_maf_rqs': NeuralInferenceFactory.create_nle_maf_rqs,
            'nle_nsf': NeuralInferenceFactory.create_nle_nsf,
            'snpe_c_maf_cnn': NeuralInferenceFactory.create_snpe_c_maf_cnn,
            'snpe_c_maf_resnet': NeuralInferenceFactory.create_snpe_c_maf_resnet,
            'snpe_c_maf_mlp_mild': NeuralInferenceFactory.create_snpe_c_maf_mlp_mild,
            'snpe_c_maf_mlp_medium': NeuralInferenceFactory.create_snpe_c_maf_mlp_medium,
            'snpe_c_maf_mlp_strong': NeuralInferenceFactory.create_snpe_c_maf_mlp_strong,
            'snpe_c_maf_mlp_aggressive': NeuralInferenceFactory.create_snpe_c_maf_mlp_aggressive,
            'snpe_c_maf_mlp_extreme': NeuralInferenceFactory.create_snpe_c_maf_mlp_extreme,
            'snpe_c_maf_mlp_extreme_com': NeuralInferenceFactory.create_snpe_c_maf_mlp_extreme_com,
            'snpe_c_maf_mlp_likebinning': NeuralInferenceFactory.create_snpe_c_maf_mlp_likebinning,
            'snpe_c_maf_mlp_likebinning_v2': NeuralInferenceFactory.create_snpe_c_maf_mlp_likebinning_v2,
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

    @staticmethod
    def create_snpe_c_mdn(prior):
        nn = posterior_nn(
            model="mdn",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_made(prior):
        nn = posterior_nn(
            model="made",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_rqs(prior):
        nn = posterior_nn(
            model="maf_rqs",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_nsf(prior):
        nn = posterior_nn(
            model="nsf",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_nle_mdn(prior):
        nn = likelihood_nn(
            model="mdn",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return NLE(prior=prior, density_estimator=nn)

    @staticmethod
    def create_nle_made(prior):
        nn = likelihood_nn(
            model="made",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return NLE(prior=prior, density_estimator=nn)

    @staticmethod
    def create_nle_maf_rqs(prior):
        nn = likelihood_nn(
            model="maf_rqs",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return NLE(prior=prior, density_estimator=nn)

    @staticmethod
    def create_nle_nsf(prior):
        nn = likelihood_nn(
            model="nsf",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            num_components=10,
        )
        return NLE(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_cnn(prior):
        embedding_net = CNNEmbedding(
            input_shape=(2448,),
            in_channels=1,
            out_channels_per_layer=None,
            num_conv_layers=2,
            num_linear_layers=2,
            num_linear_units=50,
            output_dim=74,
            kernel_size=5,
            pool_kernel_size=2,
        )

        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_resnet(prior):
        embedding_net = ResNetEmbedding1D(
            c_in=74,
            c_out=8,
            n_blocks=4,
            c_internal=64,
            c_hidden_final=128,
        )

        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_mlp_mild(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(74, 64),  # 14% compresión
            nn.ReLU(),
            nn.Linear(64, 74),  
        )
    
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_mlp_medium(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(74, 50),  # 32% compresión
            nn.ReLU(),
            nn.Linear(50, 74),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)
 
    @staticmethod
    def create_snpe_c_maf_mlp_strong(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(74, 32),  # 57% compresión
            nn.ReLU(),
            nn.Linear(32, 74),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_mlp_aggressive(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(74, 24),  # 68% compresión
            nn.ReLU(),
            nn.Linear(24, 74),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)
    
    @staticmethod
    def create_snpe_c_maf_mlp_extreme(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(74, 16),  # 78% compresión
            nn.ReLU(),
            nn.Linear(16, 74),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_mlp_extreme_com(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(74, 16),  
            nn.ReLU(),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_mlp_likebinning(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(2448, 74),  
            nn.ReLU(),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)

    @staticmethod
    def create_snpe_c_maf_mlp_likebinning_v2(prior):
        from torch import nn
        embedding_net = nn.Sequential(
            nn.Linear(2448, 256),
            nn.ReLU(),
            nn.Linear(256, 74),
        )
        
        nn = posterior_nn(
            model="maf", 
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            num_bins=10,
            embedding_net=embedding_net,
            num_components=10,
        )
        return SNPE_C(prior=prior, density_estimator=nn)


        
        

