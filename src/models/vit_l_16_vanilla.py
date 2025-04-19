"""
Vision Transformer (ViT-L/16) Vanilla Model
"""

import torch
import torch.nn as nn
from torchvision.models import ViT_L_16_Weights, vit_l_16


class ViTL16Vanilla(nn.Module):
    """
    Vanilla ViT-L/16 model from torchvision with ImageNet pretrained weights
    """

    def __init__(
        self,
        num_classes: int = 15,
        freeze_backbone: bool = False,
        demo_mode: bool = False,
    ):
        """
        Initialize the ViT-L/16 model
        Args:
            num_classes (int): Number of output classes. Defaults to 15
                (14 pathologies + 1 no pathology)
            freeze_backbone (bool): Whether to freeze the backbone model parameters
                during training. Defaults to False. When set to True will freeze
                all parameters in the ViT-L/16 model except for the classifier
                head.
            demo_mode (bool): Whether to use demo mode. Defaults to False.
                When set to True, the model keeps the original classifier head
                instead of replacing it with a new one. This is useful for
                demonstration purposes or when the number of classes is the same
                as the original model.
        """
        super(ViTL16Vanilla, self).__init__()
        self.model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        num_features = self.model.hidden_dim

        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if not demo_mode:
            # Replace the classifier if num_classes is different from ImageNet
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
