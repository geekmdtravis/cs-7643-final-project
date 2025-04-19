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

    def __init__(
        self,
        num_classes: int = 15,
        freeze_backbone: bool = False,
        demo_mode: bool = False,
    ):
        """
        Initialize the DenseNet-121 model
        Args:
            num_classes (int): Number of output classes. Defaults to 15
                (14 pathologies + 1 no pathology)
            freeze_backbone (bool): Whether to freeze the backbone model parameters
                during training. Defaults to False. When set to True will freeze
                all parameters in the DenseNet-121 model except for the classifier
                head.
            demo_mode (bool): Whether to use demo mode. Defaults to False.
                When set to True, the model keeps the original classifier head
                instead of replacing it with a new one. This is useful for
                demonstration purposes or when the number of classes is the same
                as the original model.
        """
        super(DenseNet121Vanilla, self).__init__()
        # ChestXNet used pretrained=True
        # instead of weights=DenseNet121_Weights.IMAGENET1K_V1
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.model.classifier.in_features

        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if not demo_mode:
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
