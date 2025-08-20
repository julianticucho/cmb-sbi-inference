import camb
import numpy as np
import torch
import os
from tqdm import tqdm
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.inference import simulate_for_sbi

from src.config import PARAMS, PATHS
from src.prior import get_prior
from src.bin import bin_simulations


class SpectrumSimulator:
    def __init__(self, type_str="TT+EE+BB+TE"):
        self.type_str = type_str

    def compute_spectrum(self, params):
        """Calcula el espectro teórico CMB para un conjunto de parámetros"""
        ombh2, omch2, theta_MC_100, ln_10_10_As, ns = params
        pars = camb.CAMBparams()
        pars.set_cosmology(
            ombh2=ombh2,
            omch2=omch2,
            tau=0.0522,
            cosmomc_theta=theta_MC_100 / 100,
        )
        pars.InitPower.set_params(
            ns=ns,
            As=np.exp(np.asarray(ln_10_10_As)) / 1e10
        )
        pars.set_for_lmax(2500)
        pars.set_accuracy(AccuracyBoost=1.0)
        pars.NonLinear = camb.model.NonLinear_both
        pars.WantLensing = True

        results = camb.get_results(pars)
        cmb_power_spectra = results.get_cmb_power_spectra(pars, CMB_unit="muK")["total"]

        return np.concatenate([cmb_power_spectra[:, i] for i in range(4)])

    def create_simulator(self):
        """Crea el simulador siguiendo la pipeline"""
        def simulator(theta):
            cmb_power_spectra = None

            if self.type_str == "TT+EE+BB+TE":
                cmb_power_spectra = self.compute_spectrum(theta)

            elif self.type_str == "TT":
                cmb_power_spectra = self.compute_spectrum(theta)[:2551]

            elif self.type_str == "EE":
                cmb_power_spectra = self.compute_spectrum(theta)[2551:5102]

            elif self.type_str == "BB":
                cmb_power_spectra = self.compute_spectrum(theta)[5102:7653]

            elif self.type_str == "TE":
                cmb_power_spectra = self.compute_spectrum(theta)[7653:]

            elif self.type_str == "TT+EE":
                cmb_power_spectra = self.compute_spectrum(theta)[:5102]

            elif self.type_str == "TT+EE+TE":
                cmb_power_spectra = np.concatenate([
                    self.compute_spectrum(theta)[:5102],
                    self.compute_spectrum(theta)[7653:]
                ])

            elif self.type_str == "TT+lowEE+lowTE":
                cmb_power_spectra = np.concatenate([
                    self.compute_spectrum(theta)[:2551],
                    self.compute_spectrum(theta)[2551:2582],
                    self.compute_spectrum(theta)[7653:7684]
                ])

            elif self.type_str == "TT+noise":
                cmb_power_spectra = self.compute_spectrum(theta)[:2551]
                cmb_power_spectra = NoiseModel.add_instrumental_noise_static(cmb_power_spectra)
                cmb_power_spectra = NoiseModel.sample_observed_spectra_static(cmb_power_spectra)

            elif self.type_str.endswith("bin500") or self.type_str.endswith("bin100"):
                cmb_power_spectra = self._apply_binning(theta)

            return torch.from_numpy(cmb_power_spectra)

        return simulator

    def _apply_binning(self, theta):
        """Aplica binning según el tipo_str"""
        binsize = 500 if "bin500" in self.type_str else 100

        def bin_section(spec, start, end):
            t = torch.from_numpy(spec).float().unsqueeze(0)
            return bin_simulations(t, 0, end - start, binsize)[1].squeeze(0).numpy()

        cmb_power_spectra = []
        if "TT" in self.type_str:
            cmb_power_spectra.append(bin_section(self.compute_spectrum(theta)[:2551], 0, 2550))
        if "EE" in self.type_str:
            cmb_power_spectra.append(bin_section(self.compute_spectrum(theta)[2551:5102], 0, 2550))
        if "BB" in self.type_str:
            cmb_power_spectra.append(bin_section(self.compute_spectrum(theta)[5102:7653], 0, 2550))
        if "TE" in self.type_str:
            cmb_power_spectra.append(bin_section(self.compute_spectrum(theta)[7653:], 0, 2550))

        return np.concatenate(cmb_power_spectra, axis=0)

    @staticmethod
    def Cl_XX(concatenate_batches, spectrum_type):
        """Devuelve un vector 1D de los espectros de dos puntos concatenados"""
        if spectrum_type == "TT":
            return concatenate_batches[:, :2551]
        elif spectrum_type == "EE":
            return concatenate_batches[:, 2551:5102]
        elif spectrum_type == "BB":
            return concatenate_batches[:, 5102:7653]
        elif spectrum_type == "TE":
            return concatenate_batches[:, 7653:]
        elif spectrum_type == "TT+EE":
            return concatenate_batches[:, :5102]
        elif spectrum_type == "TT+EE+TE":
            return torch.concatenate(
                (concatenate_batches[:, :5102], concatenate_batches[:, 7653:]),
                dim=1
            )
        elif spectrum_type == "TT+lowEE+lowTE":
            return torch.concatenate(
                (
                    concatenate_batches[:, :2551],
                    concatenate_batches[:, 2551:2582],
                    concatenate_batches[:, 7653:7684]
                ),
                dim=1
            )
        elif spectrum_type == "TT+EE+BB+TE":
            return concatenate_batches


class NoiseModel:
    def __init__(self, noise_config=None):
        self.noise_config = noise_config or PARAMS["noise"]

    @staticmethod
    def add_instrumental_noise_static(spectra):
        noise_config = PARAMS["noise"]
        lmax = spectra.shape[0]
        ell = np.arange(lmax)
        theta_fwhm_rad = noise_config["theta_fwhm"] * np.pi / (180 * 60)
        Nl_TT = (theta_fwhm_rad * noise_config["sigma_T"])**2 * np.exp(
            ell*(ell+1)*(theta_fwhm_rad**2)/(8*np.log(2))
        )
        return spectra + Nl_TT

    @staticmethod
    def sample_observed_spectra_static(spectra):
        noise_config = PARAMS["noise"]
        noisy_spectra = np.zeros_like(spectra)
        lmax = spectra.shape[0]

        for ell in range(2, lmax):
            C_ell = spectra[ell]
            if ell < noise_config["l_transition"]:
                dof = int(round(noise_config["f_sky"] * (2*ell + 1)))
                dof = max(dof, 1)
                samples = np.random.normal(scale=np.sqrt(C_ell), size=dof)
                noisy_spectra[ell] = np.sum(samples**2) / dof
            else:
                var = 2 * C_ell**2 / (noise_config["f_sky"] * (2*ell + 1))
                noisy_spectra[ell] = np.random.normal(loc=C_ell, scale=np.sqrt(var))
        return noisy_spectra

    def add_instrumental_noise(self, spectra):
        return NoiseModel.add_instrumental_noise_static(spectra)

    def sample_observed_spectra(self, spectra):
        return NoiseModel.sample_observed_spectra_static(spectra)

    def generate_noise(self, x):
        """Añade ruido a los Cls a partir de un tensor"""
        x_np = x.numpy() if torch.is_tensor(x) else x
        spectra_with_noise = np.array([self.add_instrumental_noise(spec) for spec in x_np])
        noisy_spectra = np.array([self.sample_observed_spectra(spec) for spec in spectra_with_noise])
        return torch.from_numpy(noisy_spectra).float()

    def generate_noise_multiple(self, x, K=10):
        """Genera K realizaciones de ruido por cada espectro"""
        x_np = x.numpy() if torch.is_tensor(x) else x
        spectra_list = []
        for spec in tqdm(x_np, desc="Generando ruido", unit="simulación"):
            for _ in range(K):
                spec_noise = self.add_instrumental_noise(spec)
                spec_noise = self.sample_observed_spectra(spec_noise)
                spectra_list.append(spec_noise)
        return torch.from_numpy(np.array(spectra_list)).float()

    def generate_noise_multiple_inplace(self, x, K=10):
        """Genera K realizaciones de ruido por espectro, en un tensor preasignado"""
        num_sims = x.shape[0]
        lmax = x.shape[1]
        spectra_with_noise = torch.empty((num_sims * K, lmax), dtype=torch.float32)

        idx = 0
        for spec in tqdm(x, desc="Generando ruido", unit="simulación"):
            for _ in range(K):
                spec_noise = self.add_instrumental_noise(spec.numpy())
                spec_noise = self.sample_observed_spectra(spec_noise)
                spectra_with_noise[idx] = torch.from_numpy(spec_noise).float()
                idx += 1
        return spectra_with_noise


class CosmologyPipeline:
    def __init__(self, type_str="TT+EE+BB+TE"):
        self.simulator = SpectrumSimulator(type_str=type_str)
        self.noise_model = NoiseModel()

    def generate_cosmologies(self, num_simulations):
        """Genera simulaciones en batches"""
        prior = get_prior()
        prior, _, prior_returns_numpy = process_prior(prior)
        simulator_wrapper = process_simulator(self.simulator.create_simulator(), prior, prior_returns_numpy)
        theta, x = simulate_for_sbi(
            simulator_wrapper,
            proposal=prior,
            num_simulations=num_simulations,
            num_workers=11,
            seed=1
        )
        return theta, x

    def save_simulations(self, theta, x, filename):
        torch.save({"theta": theta, "x": x}, os.path.join(PATHS["simulations"], filename))
        print(f"Simulaciones guardadas en {filename}")

    def load_simulations(self, filename):
        data = torch.load(os.path.join(PATHS["simulations"], filename), weights_only=True)
        return data["theta"], data["x"]
