import numpy as np
import torch
import camb
import healpy as hp
import pysm3
import pysm3.units as u
from typing import List, Optional, Dict, Tuple
from ..contracts.base_simulator import BaseSimulator


class MapBasedSimulator(BaseSimulator):    
    
    # definimos la resolucion del mapa nside, las 8 bandas de 
    # frecuencias con sus respectivos parametros instrumentales
    # y se cargan los modelos d1 y s1 para el polvo y el sincrotron
    def __init__(
        self,
        nside: int = 128,
        lmax: int = 384,
        band_params: Dict[int, Tuple[float, float, float, float]] = {
            27:  (33.0, -2.4, 15.0, 91.0),
            39:  (22.0, -2.4, 15.0, 63.0),
            93:  (2.5, -2.5, 25.0, 30.0),
            145: (2.8, -3.0, 25.0, 17.0),
            225: (5.5, -3.0, 35.0, 11.0),
            280: (14.0, -3.0, 40.0, 9.0),
        }
    ):
        super().__init__(data_type='map')
        self.nside = nside
        self.lmax = lmax
        
        # band_params: diccionario con los parametros instrumentales
        # nu: (sigma_arcmin, alpha, lknee, fwhm_arcmin)
        # units: (microK arcmin, dimensionless, dimensionless, arcmin)
        self.band_params = band_params
        
        sorted_nus = sorted(self.band_params.keys(), key=lambda nu: self.band_params[nu][3], reverse=True)
        self.frequencies = sorted_nus
        self.beam_fwhms = [self.band_params[nu][3] for nu in self.frequencies]
        self.freqs_ghz = [float(nu) for nu in self.frequencies]
        self.dust_sky = pysm3.Sky(nside=self.nside, preset_strings=["d1"])
        self.sync_sky = pysm3.Sky(nside=self.nside, preset_strings=["s1"])

    # generamos el espectro de potencia Nl para una frecuencia especificada
    # usando Nl = (sigma_arcmin * pi / 10800)**2 * (1 + (ell / lknee)**alpha)
    def _get_noise_nl(self, nu: float) -> np.ndarray:
        sigma, alpha, lknee, _ = self.band_params[nu]
        ell = np.arange(self.lmax + 1)
        w_inv = (sigma * np.pi / 10800.0)**2
        nl = np.ones(self.lmax + 1) * w_inv
        if alpha != 0:
            mask = ell > 0
            nl[mask] *= (1.0 + (ell[mask] / lknee)**alpha)
        return nl

    # generamos el espectro de potencia cosmologico Cl BB a partir de r y A_lens
    # se fijan los parametros en camb, se activa la simulacion de tensores y de 
    # lensing, se devuelven los Cls y no los Dls 
    def _get_theory_cl_bb(self, r: float, A_lens: float) -> np.ndarray:
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.32, ombh2=0.02237, omch2=0.1201, tau=0.0544)
        pars.InitPower.set_params(r=r)
        pars.Alens = A_lens
        pars.WantLensing = True
        pars.set_for_lmax(self.lmax)
        pars.WantTensors = True
        results = camb.get_results(pars)
        # camb returns Dl = l(l+1)Cl / 2pi
        dl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:, 2]
        dl = dl[:self.lmax + 1]
        ell = np.arange(len(dl))
        cl = np.zeros_like(dl)
        mask = ell > 1
        cl[mask] = dl[mask] * (2 * np.pi) / (ell[mask] * (ell[mask] + 1))
        return cl

    # transformamos los coeficientes alm de los modos E y B al espacio de pixeles
    # el arreglo de ceros se utiliza para la componente de temperatura (T)
    # y devolvemos la transformacion completa mapeada a los campos Q y U de Stokes
    def _alm2map(self, alm_e: np.ndarray, alm_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zero_alm = np.zeros_like(alm_e)
        _, q, u = hp.alm2map(
            [zero_alm, alm_e, alm_b], 
            nside=self.nside, 
            lmax=self.lmax, 
            pol=True
        )
        return q, u

    # generamos un Cl BB y sampleamos una realizacion gaussiana en espacio armonico
    # generamos los alm del modo B de forma aislada y asignamos un arreglo de ceros 
    # para el modo E ya que solo modelamos modos B
    def _generate_cmb_alms(self, r: float, A_lens: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        cl_bb = self._get_theory_cl_bb(r, A_lens)
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed)
            alm_b = hp.synalm(cl_bb, lmax=self.lmax)
            np.random.set_state(state)
        else:
            alm_b = hp.synalm(cl_bb, lmax=self.lmax)
        alm_e = np.zeros_like(alm_b)
        return alm_e, alm_b

    # el beam factor se calcula como exp(-0.5 * sigma**2 * (ell * (ell + 1) - s**2))
    # donde s = 2 para polarizacion y sigma = fwhm / (2 * sqrt(2 * log(2)))
    # devolvemos el perfil del filtro gaussiano para aplicarlo sobre los alms
    def _get_beam_factor(self, fwhm: float) -> np.ndarray:
        fwhm_rad = np.radians(fwhm / 60.0)
        sigma = fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        ell = np.arange(self.lmax + 1.0)
        beam_factor = np.exp(-0.5 * (ell * (ell + 1) - 2**2) * sigma**2)
        return beam_factor

    # generamos un Cl NL y sampleamos una realizacion gaussiana en espacio armonico
    # lo mismo que en _generate_cmb_alms pero obteniendo alms de ruido para el modo B
    def _get_noise_alms(self, nu: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        nl = self._get_noise_nl(nu)
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(seed)
            alm_b = hp.synalm(nl, lmax=self.lmax)
            np.random.set_state(state)
        else:
            alm_b = hp.synalm(nl, lmax=self.lmax)
        alm_e = np.zeros_like(alm_b) 
        return alm_e, alm_b
        
    # modificamos los indices espectrales de los modelos de PySM
    # le añadimos las unidades adimensionales que espera pySM
    def _update_foreground_parameters(self, beta_d: float, beta_s: float):
        """Update spectral indices for dust and synchrotron in PySM models."""
        self.dust_sky.components[0].spectral_index = beta_d * u.dimensionless_unscaled
        self.sync_sky.components[0].spectral_index = beta_s * u.dimensionless_unscaled

    # generamos la emision de polvo y sincrotron para una frecuencia dada
    # y los escala por las amplitudes A_d y A_s, convierte las unidades
    # nativas del mapa a temperatura termodinamica del CMB en uK_CMB
    # y transformamos los mapas Q y U resultantes de una vez al espacio armonico
    def _get_foreground_alms(self, nu: float, A_d: float, A_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate scaled foreground maps at a given frequency in uK_CMB."""
        dust_map = self.dust_sky.get_emission(nu * u.GHz).to(
            u.uK_CMB, equivalencies=u.cmb_equivalencies(nu * u.GHz)
        )
        sync_map = self.sync_sky.get_emission(nu * u.GHz).to(
            u.uK_CMB, equivalencies=u.cmb_equivalencies(nu * u.GHz)
        )
        q_fg = A_d * dust_map[1].value + A_s * sync_map[1].value
        u_fg = A_d * dust_map[2].value + A_s * sync_map[2].value
        zero_map = np.zeros_like(q_fg)
        _, alm_e_fg, alm_b_fg = hp.map2alm([zero_map, q_fg, u_fg], lmax=self.lmax, pol=True)
        return alm_e_fg, alm_b_fg

    # metodo publico que el usuario ejecutara, recibe los 6 parametros en
    # un tensor de pytorch y ejecuta toda la pipeline de simulación:
    # combinar señales y aplicar el haz directamente sobre los coeficientes alm,
    # agregar ruidos independientes y pasar a pixeles al final de la ejecucion
    def simulate(self, parameters: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        r, A_lens, A_d, beta_d, A_s, beta_s = parameters.tolist()
        # actualizamos los indices espectrales de los fg
        # generamos los alms del CMB
        self._update_foreground_parameters(beta_d, beta_s)
        alm_e_cmb, alm_b_cmb = self._generate_cmb_alms(r, A_lens, seed=seed)
        split_1_maps = []
        split_2_maps = []
        for i, nu in enumerate(self.frequencies):
            # agregamos los alms de los fg y aplicamos el beam
            # obteniendo los alms suavizados
            alm_e_fg, alm_b_fg = self._get_foreground_alms(nu, A_d, A_s)
            alm_e_sky = alm_e_cmb + alm_e_fg
            alm_b_sky = alm_b_cmb + alm_b_fg
            beam_factor = self._get_beam_factor(self.band_params[nu][3])
            alm_e_sky_smoothed = hp.almxfl(alm_e_sky, beam_factor)
            alm_b_sky_smoothed = hp.almxfl(alm_b_sky, beam_factor)
            
            # generamos dos realizaciones independientes de ruido en alms
            # y combinamos con el cielo suavizado en el espacio armonico
            n_seed = seed + 1 + 2 * i if seed is not None else None
            alm_e_n1, alm_b_n1 = self._get_noise_alms(nu, seed=n_seed)
            n_seed = seed + 2 + 2 * i if seed is not None else None
            alm_e_n2, alm_b_n2 = self._get_noise_alms(nu, seed=n_seed)
            alm_e_obs1 = alm_e_sky_smoothed + alm_e_n1
            alm_b_obs1 = alm_b_sky_smoothed + alm_b_n1
            alm_e_obs2 = alm_e_sky_smoothed + alm_e_n2
            alm_b_obs2 = alm_b_sky_smoothed + alm_b_n2
            
            # transformamos al espacio de pixeles una unica vez al final
            # y guardamos los mapas segun el split
            q_obs1, u_obs1 = self._alm2map(alm_e_obs1, alm_b_obs1)
            q_obs2, u_obs2 = self._alm2map(alm_e_obs2, alm_b_obs2)
            split_1_maps.append([q_obs1, u_obs1])
            split_2_maps.append([q_obs2, u_obs2])
        
        freq_maps_np = np.stack([split_1_maps, split_2_maps], axis=0)
        return torch.from_numpy(freq_maps_np).float()