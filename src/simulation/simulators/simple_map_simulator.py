import numpy as np
import torch
import camb
import healpy as hp
import pysm3
import pysm3.units as u
from typing import List, Optional, Dict, Tuple
from ..contracts.base_simulator import BaseSimulator


class SimpleMapSimulator(BaseSimulator):    
    
    def __init__(
        self,
        nside: int = 32,  
        lmax: int = 96, # 3 * nside
        frequencies: List[int] = [27, 39, 93, 145, 225, 280],
    ):
        super().__init__(data_type='map')
        self.nside = nside
        self.lmax = lmax
        self.frequencies = frequencies

        # inicialización de pysm3 con modelos d1 (dust) y s1 (synchrotron)
        self.dust_sky = pysm3.Sky(nside=self.nside, preset_strings=["d1"])
        self.sync_sky = pysm3.Sky(nside=self.nside, preset_strings=["s1"])

    def _get_theory_cl_bb(self, r: float) -> np.ndarray:
        """Genera el Cl BB teórico usando CAMB (A_lens fijo en 1.0)."""
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.32, ombh2=0.02237, omch2=0.1201, tau=0.0544)
        pars.InitPower.set_params(r=r)
        pars.Alens = 1.0  
        pars.WantLensing = True
        pars.set_for_lmax(self.lmax)
        pars.WantTensors = True
        
        results = camb.get_results(pars)
        dl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:, 2]
        dl = dl[:self.lmax + 1]
        
        ell = np.arange(len(dl))
        cl = np.zeros_like(dl)
        mask = ell > 1
        cl[mask] = dl[mask] * (2 * np.pi) / (ell[mask] * (ell[mask] + 1))
        return cl

    def _generate_cmb_alms(self, r: float) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una realización aleatoria del CMB en espacio armónico (Modo B)."""
        cl_bb = self._get_theory_cl_bb(r)
        alm_b = hp.synalm(cl_bb, lmax=self.lmax)
        alm_e = np.zeros_like(alm_b)
        return alm_e, alm_b

    def _update_foreground_parameters(self, beta_d: float, beta_s: float):
        """Actualiza los índices espectrales en PySM3."""
        self.dust_sky.components[0].spectral_index = beta_d * u.dimensionless_unscaled
        self.sync_sky.components[0].spectral_index = beta_s * u.dimensionless_unscaled

    def _get_foreground_alms(self, nu: float, A_d: float, A_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Genera y escala los mapas de emisión de Dust y Synchrotron en alms."""
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

    def _alm2map(self, alm_e: np.ndarray, alm_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transformación directa de armónicos a píxeles (Q y U)."""
        zero_alm = np.zeros_like(alm_e)
        _, q, u = hp.alm2map([zero_alm, alm_e, alm_b], nside=self.nside, lmax=self.lmax, pol=True)
        return q, u

    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        """Parameters: Tensor -> [r, A_d, beta_d, A_s, beta_s]"""
        r, A_d, beta_d, A_s, beta_s = parameters.tolist()
        self._update_foreground_parameters(beta_d, beta_s)
        alm_e_cmb, alm_b_cmb = self._generate_cmb_alms(r)
        output_maps = []
        for nu in self.frequencies:
            alm_e_fg, alm_b_fg = self._get_foreground_alms(nu, A_d, A_s)
            alm_e_sky = alm_e_cmb + alm_e_fg
            alm_b_sky = alm_b_cmb + alm_b_fg
            q_sky, u_sky = self._alm2map(alm_e_sky, alm_b_sky)
            output_maps.append([q_sky, u_sky])
        
        # shape final del tensor: (frecuencias, Stokes=2, pixeles)
        # para NSIDE=32 y 6 frecuencias, será: (6, 2, 12288)
        freq_maps_np = np.stack(output_maps, axis=0)
        return torch.from_numpy(freq_maps_np).float()