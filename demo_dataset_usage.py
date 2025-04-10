"""Example usage of the ChestXrayDataset and dataloaders."""

import torch

from src.data import create_dataloaders, download_dataset


def main():
    """Demonstrate dataset usage in all three modes."""
    # Get dataset paths
    file_paths = download_dataset()

    print("Creating dataloaders for all three modes...")

    # 1. Image-only mode
    train_loader, test_loader = create_dataloaders(
        file_paths, mode="image_only", batch_size=32
    )
    print("\nImage-only mode:")
    res = next(iter(train_loader))
    images: torch.Tensor = res[0]
    labels: torch.Tensor = res[1]
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # 2. Image and tabular mode
    train_loader, test_loader = create_dataloaders(
        file_paths, mode="image_and_tabular", batch_size=32
    )
    print("\nImage and tabular mode:")
    res = next(iter(train_loader))
    images: torch.Tensor = res[0]
    tabular: torch.Tensor = res[1]
    labels: torch.Tensor = res[2]
    print(f"Images shape: {images.shape}")
    print(f"Tabular shape: {tabular.shape}")
    print(f"Labels shape: {labels.shape}")

    # 3. Embedded image mode
    train_loader, test_loader = create_dataloaders(
        file_paths, mode="embedded_image", batch_size=32
    )
    print("\nEmbedded image mode:")
    res = next(iter(train_loader))
    images: torch.Tensor = res[0]
    labels: torch.Tensor = res[1]
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
