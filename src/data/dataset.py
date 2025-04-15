"""Custom dataset implementation for the NIH Chest X-ray dataset."""

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.preprocessing import create_working_tabular_df, randomize_df, set_seed


class ChestXrayDataset(Dataset):
    """
    Custom dataset for the NIH Chest X-ray dataset.
    Returns a tuple of (image, tabular_features, labels) for each item.
    """

    def __init__(
        self,
        clinical_data: Path,
        cxr_images_dir: Path,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            file_paths (FilePaths): Paths to the dataset files
            transform: Optional transform to be applied to the images
            seed (int): Random seed for reproducibility
        """

        # Set default transform to ToTensor if none provided
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.images_dir = cxr_images_dir

        # Set seed first for reproducibility
        set_seed(seed)  # Ensure reproducibility for any random operations

        # Load and preprocess tabular data
        clinical_df = pd.read_csv(clinical_data)
        _clinical_df = create_working_tabular_df(clinical_df)

        # Shuffle the dataset with the given seed
        _randomized_df = randomize_df(_clinical_df, seed=seed)
        self.tabular_df = _randomized_df

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.tabular_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            tuple: A tuple containing:
                - image: The chest X-ray image as a tensor
                - tabular_features: Patient metadata as a tensor
                - labels: Disease labels as a tensor
        """
        # Get image path and load image
        img_name = self.tabular_df.iloc[idx]["imageIndex"]
        img_path = os.path.join(self.images_dir, img_name)
        _image = Image.open(img_path).convert("RGB")
        # Always apply transform since we now have a default ToTensor
        image = self.transform(_image)

        # Get labels (filter columns that start with 'label_')
        labels = self.tabular_df.iloc[idx].filter(like="label_").values.astype(float)
        labels = torch.FloatTensor(labels)

        tabular_features = torch.FloatTensor(
            [
                self.tabular_df.iloc[idx]["patientAge"],
                self.tabular_df.iloc[idx]["patientGender"],
                self.tabular_df.iloc[idx]["viewPosition"],
                self.tabular_df.iloc[idx]["followUpNumber"],
            ]
        )

        return image, tabular_features, labels
