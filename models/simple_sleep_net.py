import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SimpleSleepNet(nn.Module):
    """
    SimpleSleepNet with 4 convolutional layers
    """
    def __init__(self, latent_dim=64, dropout=0.2):
        super(SimpleSleepNet, self).__init__()
        
        logger.info(f"Initializing SimpleSleepNet with latent_dim={latent_dim}, dropout={dropout}")
        
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=dropout)
        
        # 4 strati convoluzionali
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=8, padding=63, dilation=1, bias=False),
            nn.BatchNorm1d(32),
            Mish(),
            self.dropout,
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=4, padding=62, dilation=2, bias=False),
            nn.BatchNorm1d(64),
            Mish(),
            self.dropout,
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=2, padding=60, dilation=4, bias=False),
            nn.BatchNorm1d(128),
            Mish(),
            self.dropout,
            
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=28, dilation=4, bias=False),
            nn.BatchNorm1d(128),
            Mish(),
            self.dropout,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            Mish(),
            self.dropout
        )
        
        logger.info("SimpleSleepNet initialization complete (4 conv layers).")
        
    def forward(self, x):
        x = self.conv_path(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
