"""Vision Transformer (ViT-B/32) model with multi-layer FCN classifier that combines
image features with tabular data"""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_32_Weights, vit_b_32


class ViTB32MultiModal(nn.Module):
    """
    Modified ViT-B/32 model with multi-layer FCN classifier that combines
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
        Initialize the ViT-B/32 model with a multi-layer classifier
        Args:
            hidden_dims (tuple[int]): Hidden dimensions for the classifier
            dropout (float): Dropout rate for the classifier
            num_classes (int): Number of output classes. Defaults to 15
                (14 pathologies + 1 no pathology)
            tabular_features (int): Number of tabular features to combine with image
                features. Defaults to 4 due to four clinical features being
                present in the dataset
            freeze_backbone (bool): Whether to freeze the backbone model parameters
                during training. Defaults to False. When set to True will freeze
                all parameters in the ViT-B/32 model except for the classifier
                head.
        """
        super(ViTB32MultiModal, self).__init__()
        self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        image_features = self.model.hidden_dim
        prev_dim = image_features + tabular_features
        layers = []
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

        self.model.heads = nn.Identity()
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
