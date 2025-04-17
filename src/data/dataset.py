"""Custom dataset implementation for the NIH Chest X-ray dataset."""

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDataset(Dataset):
    """
    Custom dataset for the NIH Chest X-ray dataset.
    Returns a tuple of (image, tabular_features, labels) for each item.
    Upon initialization, the location of the images and the
    clinical data CSV file are provided and the dataset organizes
    the data accordingly, for use with the PyTorch DataLoader.
    """

    def __init__(
        self,
        clinical_data: Path,
        cxr_images_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the dataset.

        Args:
            clinical_data (Path): Path to the CSV file containing clinical data.
            cxr_images_dir (Path): Path to the directory containing CXR images.
            transform (Optional[transforms.Compose]): Optional transform to be applied
                on the images. If None, a default ToTensor transform is applied.

        """

        self.transform = transform if transform is not None else transforms.ToTensor()
        self.images_dir = cxr_images_dir

        self.tabular_df = pd.read_csv(clinical_data)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.tabular_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item (a tuple) from the dataset.

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
