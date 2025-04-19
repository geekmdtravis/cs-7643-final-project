"""DenseNet-201 model with multi-layer FCN classifier that combines
image features with tabular data"""

import torch
import torch.nn as nn
from torchvision.models import DenseNet201_Weights, densenet201


class DenseNet201MultiModal(nn.Module):
    """
    Modified DenseNet-201 model with multi-layer FCN classifier that combines
    image features with tabular data
    """

    def __init__(
        self,
        hidden_dims: tuple[int] = (512, 256, 128),
        dropout: float = 0.2,
        num_classes: int = 15,
        tabular_features: int = 4,
    ):
        """
        Initialize the DenseNet-201 model with a multi-layer classifier
        Args:
            hidden_dims (tuple[int]): Hidden dimensions for the classifier
            dropout (float): Dropout rate for the classifier
            num_classes (int): Number of output classes. Defaults to 15
                (14 pathologies + 1 no pathology)
            tabular_features (int): Number of tabular features to combine with image
                features. Defaults to 4 due to four clinical features being
                present in the dataset
        """
        super(DenseNet201MultiModal, self).__init__()
        self.model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

        # Create a new multi-layer classifier that combines image and tabular features
        layers = []
        image_features = self.model.classifier.in_features
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

        # Remove the original classifier
        self.model.classifier = nn.Identity()
        # Add new classifier
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
        # Get image features from DenseNet
        image_features = self.model(x)

        # Concatenate image features with tabular data
        combined_features = torch.cat([image_features, tabular_data], dim=1)

        # Pass through the new classifier
        return self.classifier(combined_features)
