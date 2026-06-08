from typing import Optional, List
from ..simulators.power_spectrum_simulator import PowerSpectrumSimulator
from ..simulators.power_spectrum_tau_simulator import PowerSpectrumTauSimulator
from ..simulators.power_spectrum_ombh2 import PowerSpectrumOmBh2Simulator
from ..simulators.power_spectrum_omch2 import PowerSpectrumOmCh2Simulator
from ..simulators.power_spectrum_theta_MC_100 import PowerSpectrumThetaMC100Simulator
from ..simulators.power_spectrum_ln_10_10_As import PowerSpectrumLn1010AsSimulator
from ..simulators.power_spectrum_ns import PowerSpectrumNsSimulator
from ..simulators.auxiliary_observable_simulator import AuxiliaryObservableSimulator
from ..simulators.map_based_simulator import MapBasedSimulator
from ..simulators.simple_map_simulator import SimpleMapSimulator
import torch


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
            'tt_ombh2': SimulatorFactory.create_tt_ombh2_spectrum,
            'tt_omch2': SimulatorFactory.create_tt_omch2_spectrum,
            'tt_theta_MC_100': SimulatorFactory.create_tt_theta_MC_100_spectrum,
            'tt_ln_10_10_As': SimulatorFactory.create_tt_ln_10_10_As_spectrum,
            'tt_ns': SimulatorFactory.create_tt_ns_spectrum,
            'auxiliary_observables': SimulatorFactory.create_auxiliary_observables,
            'simple_skymap': SimulatorFactory.create_simple_skymap,
            'd1s1_skymap': SimulatorFactory.create_d1s1_skymap,
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
    def create_tt_ombh2_spectrum() -> PowerSpectrumOmBh2Simulator:
        return PowerSpectrumOmBh2Simulator(components=['TT'])
    
    @staticmethod
    def create_tt_omch2_spectrum() -> PowerSpectrumOmCh2Simulator:
        return PowerSpectrumOmCh2Simulator(components=['TT'])
    
    @staticmethod
    def create_tt_theta_MC_100_spectrum() -> PowerSpectrumThetaMC100Simulator:
        return PowerSpectrumThetaMC100Simulator(components=['TT'])
    
    @staticmethod
    def create_tt_ln_10_10_As_spectrum() -> PowerSpectrumLn1010AsSimulator:
        return PowerSpectrumLn1010AsSimulator(components=['TT'])
    
    @staticmethod
    def create_tt_ns_spectrum() -> PowerSpectrumNsSimulator:
        return PowerSpectrumNsSimulator(components=['TT'])

    @staticmethod
    def create_auxiliary_observables() -> AuxiliaryObservableSimulator:
        def load_chain(
            chain_prefix: str,
            param_names: Optional[List[str]] = None,
            ignore_rows: float = 0.3,
        ) -> torch.Tensor:
            from getdist import loadMCSamples
            gds = loadMCSamples(chain_prefix, settings={"ignore_rows": ignore_rows})
            if param_names is None:
                return torch.tensor(gds.samples, dtype=torch.float32)
            name_to_index = {p.name: i for i, p in enumerate(gds.paramNames.names)}
            missing = [n for n in param_names if n not in name_to_index]
            if missing:
                raise ValueError(
                    f"Some param_names were not found in chain: {missing}. "
                    f"Available: {list(name_to_index.keys())}"
                )
            idxs = [name_to_index[n] for n in param_names]
            samples = torch.tensor(gds.samples[:, idxs], dtype=torch.float32)
            return samples

        mcmc_samples = load_chain(
            chain_prefix="results/chains/planck_tt_gaussian_run_4",
            param_names=["ombh2", "omch2", "theta_MC_100", "ln_10_10_As", "ns"],
            ignore_rows=0.3
        )  
        return AuxiliaryObservableSimulator(chain_samples=mcmc_samples)
    
    @staticmethod
    def create_custom_spectrum(components: List[str]) -> PowerSpectrumSimulator:
        return PowerSpectrumSimulator(components=components)
    
    @staticmethod
    def create_simple_skymap() -> MapBasedSimulator:
        return MapBasedSimulator(
            nside=128,
            lmax=384,
            band_params={
            27:  (33.0, -2.4, 15.0, 91.0),
            39:  (22.0, -2.4, 15.0, 63.0),
            93:  (2.5, -2.5, 25.0, 30.0),
            145: (2.8, -3.0, 25.0, 17.0),
            225: (5.5, -3.0, 35.0, 11.0),
            280: (14.0, -3.0, 40.0, 9.0),
        })
    
    @staticmethod
    def create_d1s1_skymap() -> SimpleMapSimulator:
        return SimpleMapSimulator(
            nside=32,
            lmax=96,
            frequencies=[27, 39, 93, 145, 225, 280],
        )

    



        
