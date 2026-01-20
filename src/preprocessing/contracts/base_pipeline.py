import torch
import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from ...simulation.contracts.base_simulator import BaseSimulator


class BasePipeline(ABC):
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, x_clean_single: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        pass

    def run_batch(self, x_clean: torch.Tensor) -> torch.Tensor:
        nsims = x_clean.shape[0]
        x_processed = []
        for i in tqdm.trange(nsims):
            x_processed.append(self.run(x_clean[i]))
        return torch.stack(x_processed)

    def simulate_example(
        self, 
        theta: torch.Tensor, 
        simulator: BaseSimulator, 
        seed: Optional[int] = None
        ) -> torch.Tensor:
        x_clean = simulator.simulate(theta)
        x_processed = self.run(x_clean, seed)
        return x_processed
