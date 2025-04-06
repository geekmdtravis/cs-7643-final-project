from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def create_metadata_boxes(
    features: torch.Tensor,
    box_size: int,
    margin: int,
    position: str = "right",
    image_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Create visual boxes representing tabular metadata.

    Args:
        features: Tensor of shape [F] containing [age, gender, view_position]
        box_size: Size of each metadata box in pixels
        margin: Margin between boxes in pixels
        position: Where to add metadata boxes ('top', 'bottom', 'right')
        image_size: Optional (height, width) of the main image for padding calculation

    Returns:
        Tensor of shape [C, H, W] containing the metadata boxes visualization
    """
    num_features = len(features)

    # Calculate dimensions based on position
    if position in ["top", "bottom"]:
        width = (box_size * num_features) + (margin * (num_features - 1))
        height = box_size
        if image_size:
            width = max(width, image_size[1])  # Match image width if needed
    else:  # right
        width = box_size
        height = (box_size * num_features) + (margin * (num_features - 1))
        if image_size:
            height = max(height, image_size[0])  # Match image height if needed

    # Create white background
    boxes = torch.ones((3, height + 2 * margin, width + 2 * margin)) * 255

    # Draw feature boxes
    for i, feature in enumerate(features):
        if position in ["top", "bottom"]:
            x = margin + i * (box_size + margin)
            y = margin
        else:  # right
            x = margin
            y = margin + i * (box_size + margin)

        # Age: grayscale intensity
        if i == 0:
            intensity = int((feature.item() / 100.0) * 255)  # Normalize age to 0-255
            color = torch.full((3, box_size, box_size), intensity)
        # Gender and View Position: binary black/white
        else:
            color = (
                torch.zeros(3, box_size, box_size)
                if feature.item() == 1
                else torch.ones(3, box_size, box_size) * 255
            )

        boxes[:, y : y + box_size, x : x + box_size] = color

    return boxes


def embed_metadata_in_image(
    image: torch.Tensor,
    metadata_boxes: torch.Tensor,
    position: str = "right",
) -> torch.Tensor:
    """Embed metadata boxes into the image.

    Args:
        image: Tensor of shape [C, H, W]
        metadata_boxes: Tensor of shape [C, H', W'] containing metadata visualization
        position: Where to add metadata ('top', 'bottom', 'right')

    Returns:
        Tensor of shape [C, H_new, W_new] containing image with embedded metadata
    """
    C, H, W = image.shape
    _, H_meta, W_meta = metadata_boxes.shape

    if position == "right":
        new_W = W + W_meta
        new_H = max(H, H_meta)
        combined = torch.ones((C, new_H, new_W)) * 255
        combined[:, :H, :W] = image
        combined[:, :H_meta, W:] = metadata_boxes

    elif position == "bottom":
        new_W = max(W, W_meta)
        new_H = H + H_meta
        combined = torch.ones((C, new_H, new_W)) * 255
        combined[:, :H, :W] = image
        combined[:, H:, :W_meta] = metadata_boxes

    else:  # top
        new_W = max(W, W_meta)
        new_H = H + H_meta
        combined = torch.ones((C, new_H, new_W)) * 255
        combined[:, H_meta:, :W] = image
        combined[:, :H_meta, :W_meta] = metadata_boxes

    return combined


def prepare_tabular_features_for_fusion(
    features: torch.Tensor,
    embedding_dim: int,
    dropout: float = 0.1,
) -> torch.Tensor:
    """Process tabular features for fusion with CNN features or ViT tokens.

    For CNN fusion: Projects features to match feature dimension
    For ViT: Creates a learnable embedding similar to CLS token

    Args:
        features: Tensor of shape [B, F] where B is batch size, F is number of features
        embedding_dim: Target embedding dimension
        dropout: Dropout probability

    Returns:
        Tensor of shape [B, embedding_dim] ready for fusion
    """
    batch_size = features.shape[0]

    # Simple linear projection as an example
    # In practice, this would be a learnable layer in the model
    projected = F.linear(
        features,
        torch.randn(embedding_dim, features.shape[1], device=features.device),
        torch.zeros(embedding_dim, device=features.device),
    )

    if dropout > 0:
        projected = F.dropout(projected, p=dropout, training=True)

    return projected


def prepare_vit_embedding(
    features: torch.Tensor,
    embedding_dim: int,
    num_heads: int = 12,
    dropout: float = 0.1,
) -> torch.Tensor:
    """Prepare tabular features as a special token for ViT.

    Similar to the CLS token, but specifically for tabular data.

    Args:
        features: Tensor of shape [B, F] where B is batch size, F is number of features
        embedding_dim: Embedding dimension to match ViT's hidden dimension
        num_heads: Number of attention heads in ViT
        dropout: Dropout probability

    Returns:
        Tensor of shape [B, 1, embedding_dim] ready to be prepended to ViT's sequence
    """
    batch_size = features.shape[0]

    # Project features to embedding dimension and add positional encoding
    # In practice, these would be learnable parameters in the model
    projected = F.linear(
        features,
        torch.randn(embedding_dim, features.shape[1], device=features.device),
        torch.zeros(embedding_dim, device=features.device),
    )

    # Add positional information
    pos_embedding = torch.zeros(1, embedding_dim, device=features.device)
    projected = projected + pos_embedding

    if dropout > 0:
        projected = F.dropout(projected, p=dropout, training=True)

    # Add sequence dimension
    return projected.unsqueeze(1)
