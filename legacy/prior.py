from torch import Tensor
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior
from src.config import PARAM_RANGES
import pyro.distributions as dist

def create_prior(device="cpu") -> BoxUniform:
    lows = Tensor([v[0] for v in PARAM_RANGES.values()]).to(device)
    highs = Tensor([v[1] for v in PARAM_RANGES.values()]).to(device)
    return BoxUniform(low=lows, high=highs, device=device), lows, highs

def get_prior(device="cpu", format="sbi"):
    raw_prior, lows, highs = create_prior(device=device)
    if format == "sbi":
        prior, _, _ = process_prior(raw_prior)
        return prior
    elif format == "pyro":
        pyro_prior = dist.Uniform(lows, highs).to_event(1)
        return pyro_prior
    else:
        raise ValueError("Invalid format")
