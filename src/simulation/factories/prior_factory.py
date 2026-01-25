from typing import Dict, Tuple, Any
from src.simulation.contracts.base_prior import BasePrior
from ..priors.standard_cosmology_prior import StandardCosmologyPrior

class PriorFactory:

    @staticmethod
    def get_available_configurations() -> dict[str, callable]:
        return {
            'standard': PriorFactory.create_standard_prior,
            'conservative': PriorFactory.create_conservative_prior,
            'tight': PriorFactory.create_tight_prior,
            'standard_tau': PriorFactory.create_standard_tau_prior,
            'custom': PriorFactory.create_custom_prior
        }

    @staticmethod
    def get_prior(prior_type: str) -> BasePrior:
        factory_methods = PriorFactory.get_available_configurations()
        if prior_type not in factory_methods:
            raise ValueError(f"Unknown prior type: {prior_type}. Available: {list(factory_methods.keys())}")
        return factory_methods[prior_type]()

    @staticmethod
    def create_sbi_prior(prior_type: str = "standard", device: str = "cpu") -> Any:
        factory_methods = PriorFactory.get_available_configurations()
        if prior_type not in factory_methods:
            raise ValueError(f"Unknown prior type: {prior_type}. Available: {list(factory_methods.keys())}")
        prior = factory_methods[prior_type]()
        return prior.to_sbi()
    
    @staticmethod
    def create_standard_prior() -> StandardCosmologyPrior:
        standard_ranges = {
            'ombh2': (0.02212-0.00022*5, 0.02212+0.00022*5), 
            'omch2': (0.1206-0.0021*5, 0.1206+0.0021*5), 
            'theta_MC_100': (1.04077-0.00047*5, 1.04077+0.00047*5), 
            'ln_10_10_As': (3.04-0.016*5, 3.04+0.016*5), 
            'ns': (0.9626-0.0057*5, 0.9626+0.0057*5), 
        }
        return StandardCosmologyPrior(parameter_ranges=standard_ranges)
    
    @staticmethod
    def create_conservative_prior() -> StandardCosmologyPrior:
        conservative_ranges = {
            'ombh2': (0.02212-0.00022*10, 0.02212+0.00022*10), 
            'omch2': (0.1206-0.0021*10, 0.1206+0.0021*10), 
            'theta_MC_100': (1.04077-0.00047*10, 1.04077+0.00047*10), 
            'ln_10_10_As': (3.04-0.016*10, 3.04+0.016*10), 
            'ns': (0.9626-0.0057*10, 0.9626+0.0057*10), 
        }
        return StandardCosmologyPrior(parameter_ranges=conservative_ranges)
    
    @staticmethod
    def create_tight_prior() -> StandardCosmologyPrior:
        tight_ranges = {
            'ombh2': (0.021, 0.023),
            'omch2': (0.11, 0.13),
            'theta_MC_100': (1.038, 1.043),
            'ln_10_10_As': (3.0, 3.2),
            'ns': (0.95, 0.98)
        }
        return StandardCosmologyPrior(parameter_ranges=tight_ranges)

    @staticmethod
    def create_standard_tau_prior() -> StandardCosmologyPrior:
        standard_ranges = {
            'ombh2': (0.02212-0.00022*5, 0.02212+0.00022*5), 
            'omch2': (0.1206-0.0021*5, 0.1206+0.0021*5), 
            'theta_MC_100': (1.04077-0.00047*5, 1.04077+0.00047*5), 
            'ln_10_10_As': (3.04-0.016*5, 3.04+0.016*5), 
            'ns': (0.9626-0.0057*5, 0.9626+0.0057*5),
            'tau': (0.0522-0.008*5, 0.0522+0.008*5) 
        }
        return StandardCosmologyPrior(parameter_ranges=standard_ranges)
    
    @staticmethod
    def create_custom_prior(parameter_ranges: Dict[str, Tuple[float, float]]) -> StandardCosmologyPrior:
        return StandardCosmologyPrior(parameter_ranges=parameter_ranges)
    

    


