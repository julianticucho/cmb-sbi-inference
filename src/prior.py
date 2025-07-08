from torch import Tensor
from sbi.utils import BoxUniform
from src.config import PARAM_RANGES
from sbi.utils.user_input_checks import process_prior

def create_prior(device="cpu") -> BoxUniform:
    """Genera el prior en el rango de valores especificado"""
    lows = Tensor([v[0] for v in PARAM_RANGES.values()])
    highs = Tensor([v[1] for v in PARAM_RANGES.values()])
    return BoxUniform(low=lows, high=highs, device=device)

def get_prior(device="cpu"):
    """Procesa el prior para ser utilizado en sbi"""
    raw_prior = create_prior(device=device)
    prior, _, _ = process_prior(raw_prior)
    return prior