from sbi import utils as sbi_utils
from sbi.inference import SNPE
import torch
from src.model.compressor import Compressor

model = Compressor()
model.load_state_dict(torch.load("results/compressor_model.pth"))
Cl_sims = torch.load("data/raw/cmb_sims.pt").float()
params = torch.load("data/raw/sim_params.pt").float()

prior = sbi_utils.BoxUniform(
    low=torch.tensor([0.1, 0.6], dtype=torch.float32),
    high=torch.tensor([0.5, 1.0], dtype=torch.float32)
)

inference = SNPE(prior=prior)
inference.append_simulations(
    theta=params, 
    x=Cl_sims, 
)

density_estimator = inference.train(
    max_num_epochs=100,
    stop_after_epochs=20,
    force_first_round_loss=True  
)
posterior = inference.build_posterior(density_estimator)
Cl_obs = Cl_sims[0].unsqueeze(0)  
samples = posterior.sample((10000,), x=Cl_obs)

torch.save(samples, "results/posterior_samples.pt")
print("¡Inferencia completada! Muestras guardadas en results/posterior_samples.pt")