"""Helper functions for creating dataloaders for the NIH Chest X-ray dataset."""

from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.dataset import ChestXrayDataset
from src.data.download import FilePaths


def create_dataloaders(
    file_paths: FilePaths,
    mode: Literal["image_only", "image_and_tabular", "embedded_image"] = "image_only",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for the NIH Chest X-ray dataset.

    Args:
        file_paths (FilePaths): Paths to the dataset files
        mode (str): Dataset mode. One of:
            - "image_only": Returns (image, labels)
            - "image_and_tabular": Returns (image, tabular_features, labels)
            - "embedded_image": Returns (embedded_image, labels)
        batch_size (int): Batch size for the dataloaders
        train_ratio (float): Ratio of data to use for training (0 < train_ratio < 1)
        seed (int): Random seed for reproducibility
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, test_loader)
    """

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if mode not in ["image_only", "image_and_tabular", "embedded_image"]:
        raise ValueError(
            'mode must be one of ["image_only", "image_and_tabular", "embedded_image"].'
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(file_paths, FilePaths):
        raise ValueError("file_paths must be an instance of FilePaths.")
    # Standard transforms for medical images
    # - The normalization values are based on ImageNet statistics
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    dataset = ChestXrayDataset(file_paths, mode=mode, transform=transform, seed=seed)

    # Calculate train/test sizes
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
