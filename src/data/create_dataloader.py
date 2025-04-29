"""Helper functions for creating dataloaders for the NIH Chest X-ray dataset."""

import logging
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import ChestXrayDataset

# Calculated dataset statistics (update if recalculating)
DATASET_MEAN = 0.4995
DATASET_STD = 0.2480

NormalizationMode = Literal["imagenet", "dataset_specific", "none"]


def create_dataloader(
    clinical_data: Path,
    cxr_images_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    normalization_mode: NormalizationMode = "imagenet",
) -> DataLoader:
    """
    Create a dataloader from a combination of CXR imaging data
    and tabular data.

    Args:
        clinical_data (Path): Path to the tabular data.
        cxr_images_dir (Path): Path to the directory containing CXR images.
        batch_size (int): Batch size for the dataloaders
        num_workers (int): Number of workers for data loading
        normalization_mode (NormalizationMode): Normalization mode for the images.
            Options are "imagenet", "dataset_specific", or "none".
            "imagenet" applies ImageNet normalization,
            "dataset_specific" applies dataset-specific normalization,
            and "none" applies no normalization.

    Returns:
        DataLoader: The created dataloader.
    """
    logging.debug(
        "create_dataloader: Creating dataloader with the following parameters:\n"
        "\t- clinical_data: %s\n"
        "\t- cxr_images_dir: %s\n"
        "\t- batch_size: %d\n"
        "\t- num_workers: %d\n"
        "\t- normalization_mode: %s",
        clinical_data,
        cxr_images_dir,
        batch_size,
        num_workers,
        normalization_mode,
    )
    if batch_size <= 0:
        logging.error("Batch size must be a positive integer.")
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        logging.error("Number of workers must be a non-negative integer.")
        raise ValueError("num_workers must be a non-negative integer.")

    transform_list = [transforms.ToTensor()]

    if normalization_mode == "imagenet":
        logging.info("create_dataloader: Using ImageNet normalization.")
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif normalization_mode == "dataset_specific":
        logging.info(
            "create_dataloader: Using dataset norm "
            f"(Mean: {DATASET_MEAN:.4f}, Std: {DATASET_STD:.4f})."
        )
        transform_list.append(
            transforms.Normalize(mean=[DATASET_MEAN] * 3, std=[DATASET_STD] * 3)
        )
    elif normalization_mode == "none":
        logging.info("create_dataloader: No normalization applied.")
        pass
    else:
        logging.error(
            f"create_dataloader: Unknown normalization_mode: {normalization_mode}"
        )
        raise ValueError(
            f"create_dataloader: Unknown normalization_mode: {normalization_mode}"
        )

    transform = transforms.Compose(transform_list)

    dataset = ChestXrayDataset(
        clinical_data=clinical_data,
        cxr_images_dir=cxr_images_dir,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
