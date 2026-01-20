from typing import Optional
import torch
from ...simulation.factories import SimulatorFactory
from ..factories import PipelineFactory


class ObservationFactory:

    @staticmethod
    def get_available_configurations():
        return {
            "planck_tt": ObservationFactory.create_planck_tt_observation,
            "unbinned_planck_tt": ObservationFactory.create_unbinned_planck_tt_observation,
        }

    @staticmethod
    def get_observation(obs_type: str) -> callable:
        configurations = ObservationFactory.get_available_configurations()
        if obs_type not in configurations:
            raise ValueError(f"Unknown observation type: {obs_type}. Available: {list(configurations.keys())}")
        return configurations[obs_type]()

    @staticmethod
    def create_planck_tt_observation():
        def observation(theta_true: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
            available_simulators = SimulatorFactory.get_available_configurations()
            available_pipelines = PipelineFactory.get_available_pipelines()
            
            simulator = available_simulators["tt"]()
            pipeline = available_pipelines["planck_processing"]()
            
            return pipeline.simulate_example(theta_true, simulator, seed=seed)
        
        return observation
    
    @staticmethod
    def create_unbinned_planck_tt_observation():
        def observation(theta_true: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
            available_simulators = SimulatorFactory.get_available_configurations()
            available_pipelines = PipelineFactory.get_available_pipelines()
            
            simulator = available_simulators["tt"]()
            pipeline = available_pipelines["unbinned_planck_processing"]()
            
            return pipeline.simulate_example(theta_true, simulator, seed=seed)
        
        return observation
        