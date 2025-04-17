"""
DenseNet-121 Vanilla Model
"""

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Vanilla(nn.Module):
    """
    Vanilla DenseNet-121 model from torchvision with ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = 15):
        super(DenseNet121Vanilla, self).__init__()
        # ChestXNet used pretrained=True instead of weights=DenseNet121_Weights.IMAGENET1K_V1
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.model.classifier.in_features

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
