from typing import Optional, Tuple, List

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet stats
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),  # ImageNet stats
) -> T.Compose:
    """Get training transforms with augmentation.

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization standard deviation values

    Returns:
        Composed transform pipeline
    """
    return T.Compose(
        [
            T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            T.RandomAutocontrast(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Get validation/test transforms without augmentation.

    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization standard deviation values

    Returns:
        Composed transform pipeline
    """
    return T.Compose(
        [
            T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def normalize_tabular_features(
    features: torch.Tensor,
    feature_means: Optional[torch.Tensor] = None,
    feature_stds: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Normalize tabular features using z-score normalization.

    Args:
        features: Tensor of shape [N, F] where N is batch size and F is number of features
        feature_means: Optional pre-computed means for each feature
        feature_stds: Optional pre-computed standard deviations for each feature

    Returns:
        Tuple of:
            - Normalized features tensor
            - Computed/provided means (for consistent normalization in val/test)
            - Computed/provided stds (for consistent normalization in val/test)
    """
    if feature_means is None or feature_stds is None:
        feature_means = features.mean(dim=0)
        feature_stds = features.std(dim=0)

    # Handle constant features (std = 0)
    feature_stds = torch.where(feature_stds == 0, 1.0, feature_stds)

    normalized_features = (features - feature_means) / feature_stds
    return normalized_features, feature_means, feature_stds


class TabularFeatureNormalizer:
    """Class to maintain consistent feature normalization across train/val/test."""

    def __init__(self):
        self.feature_means: Optional[torch.Tensor] = None
        self.feature_stds: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor) -> None:
        """Compute normalization parameters from training data.

        Args:
            features: Training features of shape [N, F]
        """
        self.feature_means = features.mean(dim=0)
        self.feature_stds = features.std(dim=0)
        # Handle constant features
        self.feature_stds = torch.where(self.feature_stds == 0, 1.0, self.feature_stds)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using computed parameters.

        Args:
            features: Features to normalize of shape [N, F]

        Returns:
            Normalized features
        """
        if self.feature_means is None or self.feature_stds is None:
            raise ValueError(
                "Normalizer must be fit with training data before transform can be used"
            )
        return (features - self.feature_means) / self.feature_stds

    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Compute parameters and normalize features.

        Args:
            features: Features to fit and normalize of shape [N, F]

        Returns:
            Normalized features
        """
        self.fit(features)
        return self.transform(features)


def create_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Create transforms based on split type.

    Args:
        split: One of 'train', 'val', or 'test'
        image_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization standard deviation values

    Returns:
        Composed transform pipeline
    """
    if split == "train":
        return get_train_transforms(image_size, mean, std)
    return get_val_transforms(image_size, mean, std)
