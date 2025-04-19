"""
Vision Transformer (ViT-B/32) Vanilla Model
"""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_32_Weights, vit_b_32


class ViTB32Vanilla(nn.Module):
    """
    Vanilla ViT-B/32 model from torchvision with ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = 15):
        super(ViTB32Vanilla, self).__init__()
        self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
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
