import torch
from typing import Optional
from ..contracts.base_simulator import BaseSimulator


class AuxiliaryObservableSimulator(BaseSimulator):
    """
    Simulator implementing the auxiliary-observable trick from 
    Abellán et al. 2025 (arXiv:2507.22990v1).
    """
    
    def __init__(
        self,
        chain_samples: torch.Tensor,
        seed: Optional[int] = None,
    ):
        super().__init__(data_type='auxiliary_observable')
        self.chain_samples = chain_samples
        self.n_samples, self.n_params = chain_samples.shape
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
    
    def simulate(self, parameters: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if parameters.ndim != 1:
            raise ValueError(f"Parameters must be 1D, got shape {parameters.shape}")
        
        if parameters.shape[0] != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {parameters.shape[0]}"
            )
        
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.set_state(self.rng.get_state())
        
        # sample θ' from chain (uniform random index, follows p_L(θ))
        # and compute auxiliary observable: a = θ - θ'
        idx = torch.randint(0, self.n_samples, (1,), generator=rng).item()
        theta_chain = self.chain_samples[idx]
        a = parameters - theta_chain
        
        return a
