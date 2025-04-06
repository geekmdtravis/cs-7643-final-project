#!/usr/bin/env python3

import os
import sys
import kagglehub
from pathlib import Path


def download_dataset(target_dir: str = "downloaded_data") -> None:
    """
    Download the NIH Chest X-ray dataset using kagglehub.

    Args:
        target_dir: Directory where the dataset should be stored
    """
    try:
        # Create target directory if it doesn't exist
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        print("Downloading NIH Chest X-ray dataset...")
        # Download dataset
        cache_path = kagglehub.dataset_download(
            "khanfashee/nih-chest-x-ray-14-224x224-resized", path=str(target_path)
        )

        print(f"\nDataset downloaded successfully to: {cache_path}")
        print("\nExpected files:")
        print("  - images-224/")
        print("  - Data_Entry_2017.csv")
        print("  - train_val_list_NIH.txt")
        print("  - test_list_NIH.txt")

    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Allow custom target directory as command line argument
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "downloaded_data"
    download_dataset(target_dir)
