import torch
import torch.nn as nn

class CompressionAE(nn.Module):
    def __init__(self, input_size=2401, latent_size=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_size))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, input_size))
    
    def forward(self, x):
        z = self.encoder(x)          # Compresión
        x_recon = self.decoder(z)    # Reconstrucción
        return x_recon, z