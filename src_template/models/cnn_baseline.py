from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CNNBaseline(nn.Module):
    """Baseline CNN model for chest X-ray classification using ResNet50 backbone."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (14 for ChestX-ray14)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super().__init__()

        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original FC layer

        # Add our own classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the classifier."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, num_classes] containing logits
        """
        # Extract features through backbone [B, 2048]
        features = self.backbone(x)

        # Pass through classifier [B, num_classes]
        logits = self.classifier(features)

        return logits

    def get_backbone_output_dim(self) -> int:
        """Get the dimension of the backbone's output features.

        Returns:
            Number of features output by the backbone
        """
        return self.backbone.fc.in_features if hasattr(self.backbone, "fc") else 2048

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load a state dictionary. Handles both full and backbone-only weights.

        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce matching keys
        """
        try:
            super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            # If loading just backbone weights
            if all(k.startswith("backbone.") for k in state_dict.keys()):
                backbone_state_dict = {
                    k.replace("backbone.", ""): v for k, v in state_dict.items()
                }
                self.backbone.load_state_dict(backbone_state_dict, strict)
            else:
                raise e

    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary containing the loss value
        """
        self.train()
        optimizer.zero_grad()

        inputs, targets = batch
        inputs = inputs.to(device)
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
            batch: Tuple of (inputs, targets)
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary containing the loss value
        """
        self.eval()

        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = self(inputs)
        loss = criterion(outputs, targets)

        return {
            "val_loss": loss.item(),
            "outputs": outputs.cpu(),
            "targets": targets.cpu(),
        }
