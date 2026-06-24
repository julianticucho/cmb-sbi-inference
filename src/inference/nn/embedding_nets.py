import torch
import torch.nn as nn
import torch.nn.functional as F


class SkymapEmbedding(nn.Module):
    def __init__(self, in_channels=12, embedding_dim=5):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn0 = nn.BatchNorm1d(32)
        self.pool0 = nn.MaxPool1d(4)

        self.conv1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 12, -1)
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(x)
