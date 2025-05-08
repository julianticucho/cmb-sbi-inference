from torch import Tensor
from sbi.utils import BoxUniform
from src.simulator.config import PARAM_RANGES

def get_prior() -> BoxUniform:
    """Genera el prior en el rango de valores especificado"""
    lows = Tensor([v[0] for v in PARAM_RANGES.values()])
    highs = Tensor([v[1] for v in PARAM_RANGES.values()])
    return BoxUniform(low=lows, high=highs)