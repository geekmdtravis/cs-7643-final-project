from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from .vit_baseline import ViTBaseline


class ViTEmbed(ViTBaseline):
    """Vision Transformer model that uses images with embedded tabular data (Method 1)."""

    def __init__(
        self,
        num_classes: int = 14,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        tabular_position: str = "right",
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (14 for ChestX-ray14)
            model_name: Name of the ViT model from timm
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze the backbone layers
            tabular_position: Position of embedded tabular data ('top', 'bottom', 'right')
        """
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
        self.tabular_position = tabular_position

        if not freeze_backbone:
            # Reset patch embedding layer to learn new patterns including metadata
            old_patch_embed = self.backbone.patch_embed
            new_patch_embed = type(old_patch_embed)(
                img_size=old_patch_embed.img_size,
                patch_size=old_patch_embed.patch_size,
                in_chans=3,  # Keep same number of input channels
                embed_dim=old_patch_embed.embed_dim,
                norm_layer=(
                    old_patch_embed.norm if hasattr(old_patch_embed, "norm") else None
                ),
            )

            # Initialize with pretrained weights but allow adaptation
            with torch.no_grad():
                new_patch_embed.proj.weight.copy_(old_patch_embed.proj.weight)
                if (
                    hasattr(old_patch_embed.proj, "bias")
                    and old_patch_embed.proj.bias is not None
                ):
                    new_patch_embed.proj.bias.copy_(old_patch_embed.proj.bias)

            self.backbone.patch_embed = new_patch_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        The input images are expected to already have the tabular data
        embedded as visual elements (boxes) according to self.tabular_position.

        Args:
            x: Input tensor of shape [B, C, H, W] containing images with embedded metadata

        Returns:
            Tensor of shape [B, num_classes] containing logits
        """
        # Get CLS token embeddings [B, embed_dim]
        # The patch embeddings will automatically process the embedded metadata
        # since it's part of the input image
        features = self.backbone(x)

        # Pass through classifier [B, num_classes]
        logits = self.classifier(features)

        return logits

    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Tuple of (inputs with embedded metadata, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary containing the loss value
        """
        self.train()
        optimizer.zero_grad()

        inputs, targets = batch
        inputs = inputs.to(device)  # Already contains embedded metadata
        targets = targets.to(device)

        outputs = self(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def validate_step(
        self,
        batch: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch: Tuple of (inputs with embedded metadata, targets)
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary containing the loss value and predictions
        """
        self.eval()

        inputs, targets = batch
        inputs = inputs.to(device)  # Already contains embedded metadata
        targets = targets.to(device)

        outputs = self(inputs)
        loss = criterion(outputs, targets)

        return {
            "val_loss": loss.item(),
            "outputs": outputs.cpu(),
            "targets": targets.cpu(),
        }

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights to visualize where the model focuses.

        This is mainly for visualization/interpretation purposes.
        Extracts attention weights from the last attention layer.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Optional tensor of attention weights
        """
        self.eval()
        with torch.no_grad():
            # Get backbone features including attention weights
            features = self.backbone.forward_features(x)

            # Extract attention weights from the last layer
            # Note: Implementation depends on specific ViT architecture
            # This is a placeholder - actual implementation would need to
            # access the appropriate attention layer
            return None
