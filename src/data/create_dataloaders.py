"""Helper functions for creating dataloaders for the NIH Chest X-ray dataset."""

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import ChestXrayDataset


def create_dataloader(
    clinical_data: Path,
    cxr_images_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for the NIH Chest X-ray dataset.

    Args:
        file_paths (FilePaths): Paths to the dataset files
        batch_size (int): Batch size for the dataloaders
        train_ratio (float): Ratio of data to use for training (0 < train_ratio < 1)
        seed (int): Random seed for reproducibility
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, test_loader)
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    # Standard transforms for medical images
    # - The normalization values are based on ImageNet statistics
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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
