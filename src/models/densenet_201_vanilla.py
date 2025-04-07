"""
DenseNet-201 Vanilla Model
"""

import torch
import torch.nn as nn
from torchvision.models import densenet201, DenseNet201_Weights


class DenseNet201Vanilla(nn.Module):
    """
    Vanilla DenseNet-201 model from torchvision with ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = 1000):
        super(DenseNet201Vanilla, self).__init__()
        self.model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        num_features = self.model.classifier.in_features

        # Replace the classifier if num_classes is different from ImageNet
        if num_classes != 1000:
            self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)
