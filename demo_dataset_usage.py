"""Example usage of the ChestXrayDataset and dataloaders."""

import torch

from src.data import create_dataloaders, download_dataset
from src.utils import config


def main():
    """Demonstrate dataset usage."""
    clinical_data, cxr_images_dir = download_dataset()

    print("Creating dataloaders...")

    train_loader, test_loader = create_dataloaders(
        clinical_data=clinical_data,
        cxr_images_dir=cxr_images_dir,
        batch_size=config.BATCH_SIZE,
    )
    res = next(iter(train_loader))
    images: torch.Tensor = res[0]
    tabular: torch.Tensor = res[1]
    labels: torch.Tensor = res[2]
    print(f"Train: Images shape: {images.shape}")
    print(f"Train: Tabular shape: {tabular.shape}")
    print(f"Train: Labels shape: {labels.shape}")
    res = next(iter(test_loader))
    images: torch.Tensor = res[0]
    tabular: torch.Tensor = res[1]
    labels: torch.Tensor = res[2]
    print(f"Test: Images shape: {images.shape}")
    print(f"Test: Tabular shape: {tabular.shape}")
    print(f"Test: Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
