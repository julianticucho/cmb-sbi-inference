from torch import Tensor
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior
from srcOOP.config import PARAM_RANGES

def create_prior(device="cpu") -> BoxUniform:
    """Generate the prior in the range of values specified"""
    lows = Tensor([v[0] for v in PARAM_RANGES.values()])
    highs = Tensor([v[1] for v in PARAM_RANGES.values()])
    return BoxUniform(low=lows, high=highs, device=device)

def get_prior(device="cpu"):
    """Process the prior to be used in sbi"""
    raw_prior = create_prior(device=device)
    prior, _, _ = process_prior(raw_prior)
    return prior