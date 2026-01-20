from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator
import torch


class BaseSimulator(ABC):

    def __init__(self, data_type: str):
        self.data_type = data_type
    
    @abstractmethod
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        pass

    def simulate_batch(
        self, 
        num_simulations: int, 
        prior, seed: Optional[int] = None, 
        num_workers: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_prior, _, prior_returns_numpy = process_prior(prior)
        
        def create_simulator():
            def simulator_wrapper(theta):
                return self.simulate(theta)
            return simulator_wrapper
        
        simulator_wrapper = create_simulator()
        simulation_wrapper = process_simulator(simulator_wrapper, processed_prior, prior_returns_numpy)
        
        theta, x = simulate_for_sbi(
            simulation_wrapper,
            proposal=processed_prior,
            num_simulations=num_simulations,
            num_workers=num_workers,
            seed=seed
        )
        return theta, x


        

        


        
    

