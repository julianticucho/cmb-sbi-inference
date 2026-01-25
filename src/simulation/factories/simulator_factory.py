from typing import Optional, List
from ..simulators.power_spectrum_simulator import PowerSpectrumSimulator
from ..simulators.power_spectrum_tau_simulator import PowerSpectrumTauSimulator


class SimulatorFactory:

    @staticmethod
    def get_available_configurations() -> dict[str, callable]:
        return {
            'tt': SimulatorFactory.create_tt_spectrum,
            'tt_fast': SimulatorFactory.create_tt_spectrum_fast,
            'te': SimulatorFactory.create_te_spectrum,
            'te_fast': SimulatorFactory.create_te_spectrum_fast,
            'ee': SimulatorFactory.create_ee_spectrum,
            'ee_fast': SimulatorFactory.create_ee_spectrum_fast,
            'bb': SimulatorFactory.create_bb_spectrum,
            'bb_fast': SimulatorFactory.create_bb_spectrum_fast,
            'polarization': SimulatorFactory.create_polarization_spectra,
            'polarization_fast': SimulatorFactory.create_polarization_spectra_fast,
            'all': SimulatorFactory.create_all_spectra,
            'all_fast': SimulatorFactory.create_all_spectra_fast,
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
    def create_tt_spectrum_fast() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TT'], accuracy_boost=0.5, nonlinear=False, want_lensing=False)
    
    @staticmethod
    def create_te_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TE'])

    @staticmethod
    def create_te_spectrum_fast() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TE'], accuracy_boost=0.5, nonlinear=False, want_lensing=False)
    
    @staticmethod
    def create_ee_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['EE'])

    @staticmethod
    def create_ee_spectrum_fast() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['EE'], accuracy_boost=0.5, nonlinear=False, want_lensing=False)
    
    @staticmethod
    def create_bb_spectrum() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['BB'])

    @staticmethod
    def create_bb_spectrum_fast() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['BB'], accuracy_boost=0.5, nonlinear=False, want_lensing=False)
    
    @staticmethod
    def create_polarization_spectra() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['EE', 'BB', 'TE'])

    @staticmethod
    def create_polarization_spectra_fast() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['EE', 'BB', 'TE'], accuracy_boost=0.5, nonlinear=False, want_lensing=False)
    
    @staticmethod
    def create_all_spectra() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TT', 'EE', 'BB', 'TE'])

    @staticmethod
    def create_all_spectra_fast() -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=['TT', 'EE', 'BB', 'TE'], accuracy_boost=0.5, nonlinear=False, want_lensing=False)

    @staticmethod
    def create_tt_tau_spectrum() -> PowerSpectrumTauSimulator:
        return PowerSpectrumTauSimulator(components=['TT'])
    
    @staticmethod
    def create_custom_spectrum(components: List[str]) -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=components)
    

