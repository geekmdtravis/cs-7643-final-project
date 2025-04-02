from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_baseline import ViTBaseline


class TabularTokenizer(nn.Module):
    """Tokenizer for tabular features that creates a special token."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize the tokenizer.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (should match ViT embedding dimension)
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Small transformer to process tabular data
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,  # 4 attention heads
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project to final embedding dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, input_dim]

        Returns:
            Tensor of shape [B, 1, output_dim] representing the tabular token
        """
        # Project to hidden dimension [B, hidden_dim]
        x = self.input_proj(x)

        # Add sequence dimension [B, 1, hidden_dim]
        x = x.unsqueeze(1)

        # Pass through transformer
        x = self.transformer(x)

        # Project to output dimension
        x = self.output_proj(x)
        x = self.layer_norm(x)

        return x


class ViTFusion(ViTBaseline):
    """Vision Transformer that fuses tabular data as a special token (Method 2)."""

    def __init__(
        self,
        num_classes: int = 14,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        tabular_input_dim: int = 3,  # Age, Gender, View Position
        tabular_hidden_dim: int = 64,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (14 for ChestX-ray14)
            model_name: Name of the ViT model from timm
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze the backbone layers
            tabular_input_dim: Number of tabular features
            tabular_hidden_dim: Hidden dimension for tabular tokenizer
        """
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        # Create tabular tokenizer
        self.tabular_tokenizer = TabularTokenizer(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            output_dim=self.embed_dim,  # Match ViT embedding dimension
            dropout=dropout,
        )

        # Learnable fusion weights for combining CLS and tabular tokens
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.fusion_dropout = nn.Dropout(dropout)

        # Initialize weights
        self.initialize_fusion_weights()

    def initialize_fusion_weights(self):
        """Initialize fusion weights."""
        # Initialize to give slightly more importance to image features (CLS token)
        with torch.no_grad():
            self.fusion_weights.data = F.softmax(torch.tensor([0.6, 0.4]), dim=0)

    def fuse_tokens(
        self, cls_token: torch.Tensor, tabular_token: torch.Tensor
    ) -> torch.Tensor:
        """Fuse CLS token and tabular token.

        Args:
            cls_token: Tensor of shape [B, 1, embed_dim]
            tabular_token: Tensor of shape [B, 1, embed_dim]

        Returns:
            Fused token of shape [B, embed_dim]
        """
        # Remove sequence dimension
        cls_token = cls_token.squeeze(1)
        tabular_token = tabular_token.squeeze(1)

        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted sum of tokens
        fused = weights[0] * cls_token + weights[1] * tabular_token

        return self.fusion_dropout(fused)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            image: Image tensor of shape [B, C, H, W]
            tabular: Tabular data tensor of shape [B, tabular_input_dim]

        Returns:
            Tensor of shape [B, num_classes] containing logits
        """
        # Get backbone features including CLS token [B, num_patches + 1, embed_dim]
        features = self.backbone.forward_features(image)

        # Extract CLS token [B, 1, embed_dim]
        cls_token = features[:, 0:1, :]

        # Create tabular token [B, 1, embed_dim]
        tabular_token = self.tabular_tokenizer(tabular)

        # Fuse tokens [B, embed_dim]
        fused_features = self.fuse_tokens(cls_token, tabular_token)

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
            Dictionary containing the normalized weights for CLS and tabular tokens
        """
        weights = F.softmax(self.fusion_weights, dim=0)
        return {
            "cls_weight": weights[0].item(),
            "tabular_weight": weights[1].item(),
        }

    def get_attention_maps(
        self, image: torch.Tensor, tabular: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get attention maps including the tabular token's attention.

        Args:
            image: Image tensor of shape [B, C, H, W]
            tabular: Tabular data tensor of shape [B, tabular_input_dim]

        Returns:
            Optional tensor of attention weights
        """
        self.eval()
        with torch.no_grad():
            # This would require modifications to extract attention weights
            # from the backbone's attention layers
            # Left as a TODO for now
            return None
