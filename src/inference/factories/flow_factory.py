import torch
from typing import Literal, Dict, Callable
from torch import Tensor
import torch.nn as nn
from sbi.neural_nets.estimators import NFlowsFlow
from sbi.utils.user_input_checks import check_data_device
from sbi.utils.nn_utils import get_numel

class FlowFactory:
    def __init__(self):
        pass

    def get_available_configurations() -> Dict[str, Callable]:
        return {
            "maf": FlowFactory.build_maf,
            "nsf": FlowFactory.build_nsf,
        }

    def build_maf(
        batch_x: Tensor,
        batch_y: Tensor,
        z_score_x: Literal[
            "none", "independent", "structured", "transform_to_unconstrained"
        ] = "independent",
        z_score_y: Literal[
            "none", "independent", "structured", "transform_to_unconstrained"
        ] = "independent",
        hidden_features: int = 50,
        num_transforms: int = 5,
        embedding_net: nn.Module = nn.Identity(),
        num_blocks: int = 2,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        **kwargs,
    ) -> NFlowsFlow:
        check_data_device(batch_x, batch_y)
        x_numel = get_numel(
            batch_x,
            embedding_net=None,
            warn_on_1d=True,
        )
        y_numel = get_numel(batch_y, embedding_net=embedding_net)
        
        transform_list = []
        for _ in range(num_transforms):
            block = [
                tr
            ]

    def build_nsf(
        batch_x: Tensor,
        batch_y: Tensor,
        z_score_x: Literal[
            "none", "independent", "structured", "transform_to_unconstrained"
        ] = "independent",
        z_score_y: Literal[
            "none", "independent", "structured", "transform_to_unconstrained"
        ] = "independent",
        hidden_features: int = 50,
        num_transforms: int = 5,
        embedding_net: nn.Module = nn.Identity(),
        num_blocks: int = 2,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        **kwargs,
    ) -> NFlowsFlow:
        pass


