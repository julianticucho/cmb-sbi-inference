import torch
from torch import optim, nn
from .compressor import Compressor

Cl_normalized = torch.load("data/preprocessed/Cl_normalized.pt")
params = torch.load("data/preprocessed/params.pt")

model = Compressor(input_dim=4, hidden_dim=128, output_dim=6)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(100):
    preds = model(Cl_normalized)
    loss = loss_fn(preds, params)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "data/processed/compressor_model.pth")
print("Modelo guardado en data/processed/compressor_model.pth")