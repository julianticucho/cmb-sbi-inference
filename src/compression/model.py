import torch
import torch.nn as nn

class Compressor(nn.Module):  
    def __init__(self, input_size=2401, latent_size=6, param_dim=6):
        super().__init__()
        
        self.compression_net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, latent_size)
        )

        self.A = nn.Parameter(torch.eye(param_dim, latent_size))
        self.b = nn.Parameter(torch.zeros(param_dim))
    
    def forward(self, x):
        return self.compression_net(x)  
    
    def compute_loss(self, compressed, theta):
        target = torch.matmul(theta, self.A) + self.b
        return torch.mean(torch.sum((compressed - target)**2, dim=1))
