"""
The main SimCLR module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ResNet encoder with projection head for SimCLR
class SimCLR(nn.Module):
    """
    The main SimCLR module
    """
    def __init__(self, feature_dim=128):
        """
        Constructor
        """
        super().__init__()
        
        # Use ResNet50 as the encoder backbone, adapt for 96x96 images
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        """
        Forward pass
        """
        # Get embeddings from the encoder
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        
        # Project embeddings to the space where contrastive loss is applied
        z = self.projector(h)
        
        # Normalize the projection (important for contrastive loss)
        z = F.normalize(z, dim=1)
        
        return h, z