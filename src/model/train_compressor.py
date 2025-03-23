import torch
from torch import optim, nn
from .compressor import Compressor

Cl_sims = torch.load("data/raw/cmb_angular_sims.pt").float()  # <-- ¡Convertir aquí!
params = torch.load("data/raw/angular_sim_params.pt").float()

Cl_mean = Cl_sims.mean(dim=0)
Cl_std = Cl_sims.std(dim=0)
Cl_normalized = (Cl_sims - Cl_mean) / Cl_std

model = Compressor()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(100):
    preds = model(Cl_normalized)
    loss = loss_fn(preds, params)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "results/compressor_model.pth")
print("Modelo guardado en results/compressor_model.pth")