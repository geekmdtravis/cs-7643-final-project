"""Custom dataset implementation for the NIH Chest X-ray dataset."""

import os
from typing import Literal, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.download import FilePaths
from src.utils.image_manipulation import embed_clinical_data_into_image
from src.utils.preprocessing import create_working_tabular_df, randomize_df, set_seed


class ChestXrayDataset(Dataset):
    """
    Custom dataset for the NIH Chest X-ray dataset that supports three modes:
    1. image_only: Returns (image, labels)
    2. image_and_tabular: Returns (image, tabular_features, labels)
    3. embedded_image: Returns (embedded_image, labels)
    """

    def __init__(
        self,
        file_paths: FilePaths,
        mode: Literal[
            "image_only", "image_and_tabular", "embedded_image"
        ] = "image_only",
        transform=None,
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            file_paths (FilePaths): Paths to the dataset files
            mode (str): Dataset mode. One of:
                - "image_only": Returns (image, labels)
                - "image_and_tabular": Returns (image, tabular_features, labels)
                - "embedded_image": Returns (embedded_image, labels)
            transform: Optional transform to be applied to the images
            seed (int): Random seed for reproducibility
        """
        if mode not in ["image_only", "image_and_tabular", "embedded_image"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: "
                "'image_only', 'image_and_tabular', 'embedded_image'"
            )

        self.mode = mode
        self.transform = transform
        self.images_dir = file_paths.images_dir

        # Load and preprocess tabular data
        clinical_df = pd.read_csv(file_paths.clinical_data)
        self.tabular_df = create_working_tabular_df(clinical_df)

        # Shuffle the dataset with the given seed
        self.tabular_df = randomize_df(self.tabular_df)
        set_seed(seed)  # Ensure reproducibility for any random operations

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.tabular_df)

    def __getitem__(
        self, idx: int
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            tuple: Depending on the mode:
                - image_only: (image, labels)
                - image_and_tabular: (image, tabular_features, labels)
                - embedded_image: (embedded_image, labels)
        """
        # Get image path and load image
        img_name = self.tabular_df.iloc[idx]["imageIndex"]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get labels (filter columns that start with 'label_')
        labels = self.tabular_df.iloc[idx].filter(like="label_").values.astype(float)
        labels = torch.FloatTensor(labels)

        if self.mode == "image_only":
            return image, labels

        # Get tabular features
        tabular_features = torch.FloatTensor(
            [
                self.tabular_df.iloc[idx]["patientAge"],
                self.tabular_df.iloc[idx]["patientGender"],
                self.tabular_df.iloc[idx]["viewPosition"],
                self.tabular_df.iloc[idx]["followUpNumber"],
            ]
        )

        if self.mode == "image_and_tabular":
            return image, tabular_features, labels

        # Embedded mode
        embedded_image = embed_clinical_data_into_image(
            image,
            age=self.tabular_df.iloc[idx]["patientAge"],
            gender=(
                "female" if self.tabular_df.iloc[idx]["patientGender"] == 1 else "male"
            ),
            xr_pos="PA" if self.tabular_df.iloc[idx]["viewPosition"] == 0 else "AP",
            xr_count=self.tabular_df.iloc[idx]["followUpNumber"],
        )
        return embedded_image, labels
