from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from .cnn_baseline import CNNBaseline


class CNNEmbed(CNNBaseline):
    """CNN model that uses images with embedded tabular data (Method 1)."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        tabular_position: str = "right",
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (14 for ChestX-ray14)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
            tabular_position: Position of embedded tabular data ('top', 'bottom', 'right')
        """
        super().__init__(num_classes, pretrained, freeze_backbone)
        self.tabular_position = tabular_position

        # The first convolutional layer of ResNet might need adjustment
        # if the tabular data significantly changes the input statistics
        if not freeze_backbone:
            # Reset the first conv layer to learn new patterns including metadata boxes
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                3,  # Keep same number of input channels
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False if old_conv.bias is None else True,
            )

            # Initialize with the pretrained weights but allow for adaptation
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            self.backbone.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        The input images are expected to already have the tabular data
        embedded as visual elements (boxes) according to self.tabular_position.

        Args:
            x: Input tensor of shape [B, C, H, W] containing images with embedded metadata

        Returns:
            Tensor of shape [B, num_classes] containing logits
        """
        # Extract features through backbone [B, 2048]
        # The backbone will automatically process the embedded metadata
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

    def get_attention_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention maps to visualize where the model focuses.

        This is mainly for visualization/interpretation purposes.
        Uses Grad-CAM to create attention heatmaps.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Optional tensor of attention maps
        """
        # Note: This would require implementing Grad-CAM
        # Left as a TODO for now
        return None
