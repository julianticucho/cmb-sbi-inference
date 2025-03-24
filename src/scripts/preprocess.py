import torch

Cl_sims = torch.load("data/raw/cmb_angular_sims.pt").float()
params = torch.load("data/raw/angular_sim_params.pt").float()

Cl_sims = Cl_sims.mean(dim=1)  
Cl_mean = Cl_sims.mean(dim=0)
Cl_std = Cl_sims.std(dim=0)
Cl_normalized = (Cl_sims - Cl_mean) / Cl_std

torch.save(Cl_normalized, "data/preprocessed/Cl_normalized.pt")
torch.save(params, "data/preprocessed/params.pt")