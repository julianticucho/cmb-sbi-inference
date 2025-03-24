import torch.nn as nn

class Compressor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=6):
        super(Compressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)