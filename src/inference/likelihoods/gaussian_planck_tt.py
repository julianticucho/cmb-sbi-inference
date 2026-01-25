import torch
import numpy as np
from typing import Any, Dict
from cobaya.likelihood import Likelihood

from ...preprocessing.factories import ObservationFactory, PipelineFactory
from ...data.planck import PlanckDataLoader


class GaussianPlanckTTLikelihood(Likelihood):

    params = {
        "theta_MC_100": {"derived": True, "latex": r"100\theta_{\rm MC}"},
        "ln_10_10_As": {"derived": True, "latex": r"\ln(10^{10} A_s)"},
    }

    def initialize(self):
        self.cov_matrix, self.lmin, self.lmax, _, _, _, _, _, _ = PlanckDataLoader.load_planck_data()
        self.cov_matrix = self.cov_matrix.numpy()
        self.inv_cov = np.linalg.inv(self.cov_matrix)

        theta_fid = torch.tensor([
            0.02212,    # ombh2
            0.1206,     # omch2
            1.04077,    # theta_MC_100
            3.04,       # ln_10_10_As
            0.9626      # ns
        ])
        obs_fn = ObservationFactory.create_planck_tt_observation()
        self.x_obs = obs_fn(theta_fid, seed=0).numpy() 
        self.binning_pipeline = PipelineFactory.create_planck_binning()

        self.log.info("GaussianPlanckTTLikelihood initialized")
        self.log.info(f"Observation vector length: {len(self.x_obs)}")
        self.log.info(f"Covariance matrix shape: {self.cov_matrix.shape}")

    def get_requirements(self):
        return {'Cl': {'tt': 2500}}

    def logp(self, _derived=None, **params_values):
        if _derived is not None:
            cosmomc_theta = self.provider.get_param("cosmomc_theta")
            As = self.provider.get_param("As")
            _derived["theta_MC_100"] = float(cosmomc_theta) * 100.0
            _derived["ln_10_10_As"] = float(np.log(float(As) * 1e10))

        # get CAMB theory spectrum from Cobaya's provider
        # ell_factor=True converts C_ell to D_ell = ell(ell+1)C_ell/(2π)
        theory_cl = self.provider.get_Cl(ell_factor=True)
        tt_theory = theory_cl['tt']  

        # convert to torch and apply Planck binning
        # then compute chi2 = (x - x_obs)^T C^{-1} (x - x_obs)
        x_theory_full = torch.from_numpy(tt_theory).float()
        x_theory_binned = self.binning_pipeline.run(x_theory_full).numpy()
        diff = x_theory_binned - self.x_obs
        chi2 = diff @ self.inv_cov @ diff

        return -0.5 * chi2
