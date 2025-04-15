"""Helper functions for creating dataloaders for the NIH Chest X-ray dataset."""

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import ChestXrayDataset

# Calculated dataset statistics (update if recalculating)
DATASET_MEAN = 0.4995
DATASET_STD = 0.2480

# Define the type for normalization mode
NormalizationMode = Literal["imagenet", "dataset_specific", "none"]


def create_dataloader(
    clinical_data: Path,
    cxr_images_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    normalization_mode: NormalizationMode = "imagenet",
) -> DataLoader:  # Return type is just one DataLoader now
    """
    Create train and test dataloaders for the NIH Chest X-ray dataset.

    Args:
        file_paths (FilePaths): Paths to the dataset files
        batch_size (int): Batch size for the dataloaders
        train_ratio (float): Ratio of data to use for training (0 < train_ratio < 1)
        seed (int): Random seed for reproducibility
        num_workers (int): Number of workers for data loading

    Returns:
        DataLoader: The created dataloader.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    # Base transform
    transform_list = [transforms.ToTensor()]

    # Add normalization based on mode
    if normalization_mode == "imagenet":
        print("Using ImageNet normalization.")
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif normalization_mode == "dataset_specific":
        print(f"Using dataset norm (Mean: {DATASET_MEAN:.4f}, Std: {DATASET_STD:.4f}).")
        # Apply the same mean/std to all 3 channels since the dataset converts to RGB
        transform_list.append(
            transforms.Normalize(mean=[DATASET_MEAN] * 3, std=[DATASET_STD] * 3)
        )
    elif normalization_mode == "none":
        print("No normalization applied.")
        pass  # Only ToTensor is used
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

    transform = transforms.Compose(transform_list)

    # Create dataset
    dataset = ChestXrayDataset(
        clinical_data=clinical_data,
        cxr_images_dir=cxr_images_dir,
        transform=transform,
    )

    # Create dataloaders
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
