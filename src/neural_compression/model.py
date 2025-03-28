import torch
import torch.nn as nn

class CosmologicalNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, dropout_rate):
        super(CosmologicalNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_units, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)