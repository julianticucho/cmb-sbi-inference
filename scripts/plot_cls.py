import torch
from src.data import PlanckDataLoader
from src.visualization import plot_cl
from src.core import storage
from src.simulation.api import simulate_observation

if __name__ == "__main__":
    data_loader = PlanckDataLoader()
    cov_matrix, lmin, lmax, _, _, _, ell_high, _, err_high = data_loader.load_planck_data()

    theta_true = torch.tensor([
        0.02212,    # ombh2
        0.1206,     # omch2
        1.04077,    # theta_MC_100
        3.04,       # ln_10_10_As
        0.9626      # ns
    ])

    x_obs = simulate_observation(
        theta_true=theta_true,
        observation_type="planck_tt",
        seed=0
    )

    x_obs_clean = simulate_observation(
        theta_true=theta_true,
        observation_type="camb_tt",
        seed=0
    )

    fig = plot_cl(
        cl=x_obs,
        ell=ell_high,
        ell_min=lmin,
        ell_max=lmax,
        cl_err=err_high,
        cl_clean=x_obs_clean
    )
    storage.save_figure(fig, "cl.pdf", category="plots")
