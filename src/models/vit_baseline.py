from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import timm


class ViTBaseline(nn.Module):
    """Baseline Vision Transformer model for chest X-ray classification."""

    def __init__(
        self,
        num_classes: int = 14,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (14 for ChestX-ray14)
            model_name: Name of the ViT model from timm
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze the backbone layers
        """
        super().__init__()

        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get embedding dimension from the model
        self.embed_dim = self.backbone.embed_dim

        # Add our own classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the classifier."""
        for m in self.classifier.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, num_classes] containing logits
        """
        # Get CLS token embeddings [B, embed_dim]
        features = self.backbone(x)

        # Pass through classifier [B, num_classes]
        logits = self.classifier(features)

        return logits

    def get_backbone_output_dim(self) -> int:
        """Get the dimension of the backbone's output features.

        Returns:
            Number of features output by the backbone
        """
        return self.embed_dim

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
            Dictionary containing the loss value and predictions
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
