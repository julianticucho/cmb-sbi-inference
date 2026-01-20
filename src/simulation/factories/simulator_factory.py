from typing import Optional, List
from ..simulators.power_spectrum_simulator import PowerSpectrumSimulator
from ..simulators.power_spectrum_tau_simulator import PowerSpectrumTauSimulator


class SimulatorFactory:

    @staticmethod
    def get_available_configurations() -> dict[str, callable]:
        return {
            'tt': SimulatorFactory.create_tt_spectrum,
            'te': SimulatorFactory.create_te_spectrum,
            'ee': SimulatorFactory.create_ee_spectrum,
            'bb': SimulatorFactory.create_bb_spectrum,
            'polarization': SimulatorFactory.create_polarization_spectra,
            'all': SimulatorFactory.create_all_spectra,
            'tt_tau': SimulatorFactory.create_tt_tau_spectrum,
            'custom': SimulatorFactory.create_custom_spectrum
        }

    @staticmethod
    def get_simulator(simulator_type: str):
        simulators = SimulatorFactory.get_available_configurations()
        if simulator_type not in simulators:
            raise ValueError(f"Simulator {simulator_type} not found, available simulators: {list(simulators.keys())}")
        return simulators[simulator_type]()
    
    @staticmethod
    def create_tt_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TT'])
    
    @staticmethod
    def create_te_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TE'])
    
    @staticmethod
    def create_ee_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['EE'])
    
    @staticmethod
    def create_bb_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['BB'])
    
    @staticmethod
    def create_polarization_spectra() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['EE', 'BB', 'TE'])
    
    @staticmethod
    def create_all_spectra() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TT', 'EE', 'BB', 'TE'])

    @staticmethod
    def create_tt_tau_spectrum() -> PowerSpectrumTauSimulator:
        return PowerSpectrumTauSimulator(components=['TT'])
    
    @staticmethod
    def create_custom_spectrum(components: List[str]) -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=components)
    

