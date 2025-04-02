from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_baseline import CNNBaseline


class TabularEncoder(nn.Module):
    """Encoder for tabular features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize the encoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (should match CNN feature dimension)
            dropout: Dropout probability
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, input_dim]

        Returns:
            Tensor of shape [B, output_dim]
        """
        x = self.encoder(x)
        x = self.layer_norm(x)
        return x


class CNNFusion(CNNBaseline):
    """CNN model that fuses tabular data with image features (Method 2)."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        tabular_input_dim: int = 3,  # Age, Gender, View Position
        tabular_hidden_dim: int = 64,
        fusion_dropout: float = 0.1,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (14 for ChestX-ray14)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
            tabular_input_dim: Number of tabular features
            tabular_hidden_dim: Hidden dimension for tabular encoder
            fusion_dropout: Dropout probability for fusion layer
        """
        super().__init__(num_classes, pretrained, freeze_backbone)

        # Get CNN feature dimension
        cnn_feature_dim = self.get_backbone_output_dim()

        # Create tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            output_dim=cnn_feature_dim,
            dropout=fusion_dropout,
        )

        # Create fusion layer (simple addition with learnable weights)
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.fusion_dropout = nn.Dropout(fusion_dropout)

        # Replace classifier to handle fused features
        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_feature_dim),
            nn.Dropout(fusion_dropout),
            nn.Linear(cnn_feature_dim, num_classes),
        )

        self.initialize_fusion_weights()

    def initialize_fusion_weights(self):
        """Initialize fusion weights and classifier."""
        # Initialize fusion weights to give slightly more importance to image features
        with torch.no_grad():
            self.fusion_weights.data = F.softmax(torch.tensor([0.6, 0.4]), dim=0)

        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fuse_features(
        self, image_features: torch.Tensor, tabular_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse image and tabular features.

        Args:
            image_features: Tensor of shape [B, feature_dim]
            tabular_features: Tensor of shape [B, feature_dim]

        Returns:
            Fused features tensor of shape [B, feature_dim]
        """
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted sum of features
        fused = weights[0] * image_features + weights[1] * tabular_features

        return self.fusion_dropout(fused)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            image: Image tensor of shape [B, C, H, W]
            tabular: Tabular data tensor of shape [B, tabular_input_dim]

        Returns:
            Tensor of shape [B, num_classes] containing logits
        """
        # Get image features from CNN backbone [B, feature_dim]
        image_features = self.backbone(image)

        # Encode tabular features [B, feature_dim]
        tabular_features = self.tabular_encoder(tabular)

        # Fuse features [B, feature_dim]
        fused_features = self.fuse_features(image_features, tabular_features)

        # Pass through classifier [B, num_classes]
        logits = self.classifier(fused_features)

        return logits

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Tuple of (images, tabular_data, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary containing the loss value
        """
        self.train()
        optimizer.zero_grad()

        images, tabular, targets = batch
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        outputs = self(images, tabular)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch: Tuple of (images, tabular_data, targets)
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary containing the loss value and predictions
        """
        self.eval()

        images, tabular, targets = batch
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        outputs = self(images, tabular)
        loss = criterion(outputs, targets)

        return {
            "val_loss": loss.item(),
            "outputs": outputs.cpu(),
            "targets": targets.cpu(),
        }

    def get_fusion_weights(self) -> Dict[str, float]:
        """Get the current fusion weights.

        Returns:
            Dictionary containing the normalized weights for image and tabular features
        """
        weights = F.softmax(self.fusion_weights, dim=0)
        return {
            "image_weight": weights[0].item(),
            "tabular_weight": weights[1].item(),
        }
