"""
DenseNet-121 model with multi-layer FCN classifier that combines
image features with tabular data
"""

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121MultiModal(nn.Module):
    """
    Modified DenseNet-121 model with multi-layer FCN classifier that combines
    image features with tabular data
    """

    def __init__(
        self,
        hidden_dims: tuple[int] = (512, 256, 128),
        dropout: float = 0.2,
        num_classes: int = 15,
        tabular_features: int = 4,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the DenseNet-121 model with a multi-layer classifier
        Args:
            hidden_dims (tuple[int]): Hidden dimensions for the classifier
            dropout (float): Dropout rate for the classifier
            num_classes (int): Number of output classes. Defaults to 15
            tabular_features (int): Number of tabular features to combine with image
                features. Defaults to 4 due to four clinical features being
                present in the dataset
            freeze_backbone (bool): Whether to freeze the backbone model parameters
                during training. Defaults to False. When set to True will freeze
                all parameters in the DenseNet-121 model except for the classifier
                head.
        """
        super(DenseNet121MultiModal, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        image_features = self.model.classifier.in_features

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = nn.Identity()
        layers = []
        prev_dim = image_features + tabular_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Args:
            x (torch.Tensor): Input image tensor
            tabular_data (torch.Tensor): Input tabular data tensor
        Returns:
            torch.Tensor: Output tensor
        """
        image_features = self.model(x)
        combined_features = torch.cat([image_features, tabular_data], dim=1)
        return self.classifier(combined_features)
