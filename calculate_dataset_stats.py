import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ChestXrayDataset
from src.utils import Config

cfg = Config()


def calculate_stats(loader: DataLoader, num_samples: int) -> tuple[float, float]:
    """Calculates mean and std deviation for a dataset via DataLoader."""
    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])
    count = 0

    print("Calculating dataset statistics...")
    for images, _, _ in tqdm(loader, total=len(loader)):
        # Images are expected to be [B, C, H, W]
        # Convert to grayscale for calculation (mean across identical RGB channels)
        gray_images = images.mean(dim=1)  # Shape [B, H, W]

        # Flatten images and sum pixel values
        batch_sum = torch.sum(gray_images)
        batch_sum_sq = torch.sum(gray_images**2)

        psum += batch_sum
        psum_sq += batch_sum_sq
        count += np.prod(gray_images.shape)  # B * H * W

    # Calculate mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    return total_mean.item(), total_std.item()


def main():
    print("Setting up dataset for statistics calculation...")

    dataset = ChestXrayDataset(
        clinical_data=cfg.tabular_clinical_train,
        cxr_images_dir=cfg.cxr_train_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    mean, std = calculate_stats(loader, len(dataset))

    print("\nDataset Statistics (Grayscale):")
    print(f"Mean: {mean:.4f}")
    print(f"Std Dev: {std:.4f}")


if __name__ == "__main__":
    main()
