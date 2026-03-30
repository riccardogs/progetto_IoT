import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


#--- Added: Definition of the Mish activation function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
    
class SimpleSleepNet(nn.Module):
    """
    SimpleSleepNet is a neural network model designed for sleep stage classification using 1D convolutional layers.
    Args:
        latent_dim (int): The dimension of the latent space for the fully connected layer. Default is 128.
        dropout (float): The dropout rate to be applied after each layer. Default is 0.5.
    Attributes:
        latent_dim (int): The dimension of the latent space.
        dropout (nn.Dropout): Dropout layer.
        conv_path (nn.Sequential): Sequential container for the convolutional layers.
        fc (nn.Sequential): Sequential container for the fully connected layer.
    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor of shape (Batch, Channels, Length).
            Returns:
                torch.Tensor: Output tensor of shape (Batch, latent_dim).
    """
    def __init__(self, latent_dim=128, dropout=0.2):
        super(SimpleSleepNet, self).__init__()
        
        logger.info(f"Initializing SimpleSleepNet with latent_dim={latent_dim} and dropout={dropout}")
        
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=dropout)
        
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=8, padding=63, dilation=1, bias=False),
            nn.BatchNorm1d(32),
            Mish(),
            self.dropout,
            # First Convolutional Block: Convolution + BatchNorm + ReLU + Dropout
            # nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=8, padding=32, bias=False),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # self.dropout,
            
            # stride=4: il kernel si sposta di 4 campioni alla volta (downsampling x4)
            # dilation=2: il kernel salta un campione su due, raddoppiando il campo recettivo senza aggiungere parametri
            # padding=62: aggiunge zeri ai bordi per compensare la perdita di informazioni causata da dilation e kernel_size=32


            # Second Convolutional Block: Convolution + BatchNorm + ReLU + Dropout
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=4, padding=62, dilation=2, bias=False),
            nn.BatchNorm1d(64),
            Mish(),
            self.dropout,
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=4, padding=16, bias=False),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # self.dropout,
            
            # Third Convolutional Block: Convolution + BatchNorm + ReLU + Dropout
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=2, padding=60, dilation=4, bias=False),
            nn.BatchNorm1d(128),
            Mish(),
            self.dropout,
            # nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=2, padding=8, bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # self.dropout
        )
        
        # Fully Connected Layer for Embedding: Linear + BatchNorm + ReLU + Dropout
        self.fc = nn.Sequential(
            nn.Linear(128, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            Mish(),
            # nn.ReLU(),
            self.dropout
        )
        
        logger.info("SimpleSleepNet initialization complete.")
        
    def forward(self, x):
        logger.debug("Starting forward pass.")
        x = self.conv_path(x)          # Apply convolutional path (Batch, 128, L')
        x = F.adaptive_avg_pool1d(x, 1)  # Global average pooling (Batch, 128, 1)
        x = x.view(x.size(0), -1)     # Flatten the tensor (Batch, 128)
        x = self.fc(x)                 # Apply fully connected layer (Batch, latent_dim)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings to unit length
        logger.debug("Forward pass complete.")
        return x