"""
Vision Transformer (ViT-B/16) Vanilla Model
"""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViTB16Vanilla(nn.Module):
    """
    Vanilla ViT-B/16 model from torchvision with ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = 15):
        super(ViTB16Vanilla, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.model.hidden_dim

        if num_classes != 1000:
            self.model.heads = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)
