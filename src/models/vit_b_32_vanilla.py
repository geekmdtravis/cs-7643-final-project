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

    def __init__(
        self,
        hidden_dims: tuple[int] = (512, 256, 128),
        dropout: float = 0.2,
        num_classes: int = 15,
        freeze_backbone: bool = False,
        demo_mode: bool = False,
    ):
        """
        Initialize the ViT-B/132 model
        Args:
            hidden_dims (tuple[int]): Hidden dimensions for the classifier
            dropout (float): Dropout rate for the classifier
            num_classes (int): Number of output classes. Defaults to 15
                (14 pathologies + 1 no pathology)
            freeze_backbone (bool): Whether to freeze the backbone model parameters
                during training. Defaults to False. When set to True will freeze
                all parameters in the ViT-B/32 model except for the classifier
                head.
            demo_mode (bool): Whether to use demo mode. Defaults to False.
                When set to True, the model keeps the original classifier head
                instead of replacing it with a new one. This is useful for
                demonstration purposes or when the number of classes is the same
                as the original model.
        """
        super(ViTB32Vanilla, self).__init__()
        self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if not demo_mode:
            num_features = self.model.hidden_dim
            self.model.heads = nn.Identity()
            layers = []
            input_dim = num_features
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(input_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, num_classes))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        features = self.model(x)
        return self.classifier(features)
