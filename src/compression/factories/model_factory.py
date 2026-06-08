from typing import Callable, Dict
from ..contracts.base_model import BaseModel
from ..models.mlp import MLP


class ModelFactory:
    @staticmethod
    def get_available_configurations() -> Dict[str, Callable]:
        return {
            "mlp_default": ModelFactory.create_mlp_default,
            "mlp_unbinned": ModelFactory.create_mlp_unbinned,
            "mlp_unbinned_2layers": ModelFactory.create_mlp_unbinned_2layers,
            "mlp_200": ModelFactory.create_mlp_200
        }

    @staticmethod
    def get_model(model_type: str) -> BaseModel:
        configs = ModelFactory.get_available_configurations()
        if model_type not in configs:
            raise ValueError(
                f"Model '{model_type}' not found. "
                f"Available: {list(configs.keys())}"
            )
        return configs[model_type]()

    @staticmethod
    def create_mlp_default() -> BaseModel:
        return MLP(
            input_dim=74,
            output_dim=5,
            hidden_dims=[],
        )

    @staticmethod
    def create_mlp_unbinned() -> BaseModel:
        return MLP(
            input_dim=2448,
            output_dim=5,
            hidden_dims=[512],
        )
    
    @staticmethod
    def create_mlp_unbinned_2layers() -> BaseModel:
        return MLP(
            input_dim=2448,
            output_dim=5,
            hidden_dims=[512, 32],
        )

    @staticmethod
    def create_mlp_200() -> BaseModel:
        return MLP(
            input_dim=200,
            output_dim=5,
            hidden_dims=[64],
        )
