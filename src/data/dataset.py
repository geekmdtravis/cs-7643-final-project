from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer


class ChestXrayDataset(Dataset):
    """Base dataset class for ChestX-ray14 with tabular data, using official splits."""

    def __init__(
        self,
        image_dir: Path,
        metadata_path: Path,
        split_file_path: Path,
        split: Literal["train", "val", "test"],  # Used for transform selection
        transform=None,
        target_transform=None,
        mlb: Optional[MultiLabelBinarizer] = None,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # Load metadata
        all_metadata = pd.read_csv(metadata_path)

        # Load image filenames for the specified split
        with open(split_file_path, "r") as f:
            split_filenames = set(f.read().splitlines())

        # Filter metadata to include only images in the current split
        self.metadata = all_metadata[
            all_metadata["Image Index"].isin(split_filenames)
        ].reset_index(drop=True)

        # Preprocess metadata
        self._preprocess_metadata()

        # Prepare labels
        self.labels = self.metadata["Finding Labels"].str.split("|").tolist()
        if mlb is None:
            # Fit MultiLabelBinarizer on the training data labels if not provided
            if split == "train":
                self.mlb = MultiLabelBinarizer()
                self.mlb.fit(self.labels)
            else:
                raise ValueError(
                    "MultiLabelBinarizer must be provided for val/test splits or fit on train split."
                )
        else:
            self.mlb = mlb

        self.encoded_labels = self.mlb.transform(self.labels)
        self.classes = self.mlb.classes_

    def _preprocess_metadata(self):
        """Preprocess age, gender, and view position."""
        # Convert age to float, handle any non-numeric values, fill NaNs (e.g., with mean)
        self.metadata["Patient Age"] = pd.to_numeric(
            self.metadata["Patient Age"], errors="coerce"
        )
        # Simple NaN filling - consider more sophisticated methods if needed
        self.metadata["Patient Age"].fillna(
            self.metadata["Patient Age"].mean(), inplace=True
        )

        # Encode categorical variables, handle potential NaNs
        self.metadata["Patient Gender"] = self.metadata["Patient Gender"].map(
            {"M": 0.0, "F": 1.0}
        )
        self.metadata["Patient Gender"].fillna(
            0.5, inplace=True
        )  # Fill NaN with intermediate value

        self.metadata["View Position"] = self.metadata["View Position"].map(
            {"PA": 0.0, "AP": 1.0}
        )
        self.metadata["View Position"].fillna(
            0.5, inplace=True
        )  # Fill NaN with intermediate value

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_image(self, idx: int) -> Image.Image:
        img_path = self.image_dir / self.metadata.iloc[idx]["Image Index"]
        return Image.open(img_path).convert("RGB")

    def _get_tabular_features(self, idx: int) -> torch.Tensor:
        """Get normalized tabular features for the given index."""
        row = self.metadata.iloc[idx]
        # Ensure features are always float32
        features = torch.tensor(
            [
                float(row["Patient Age"]),
                float(row["Patient Gender"]),
                float(row["View Position"]),
            ],
            dtype=torch.float32,
        )
        # Handle potential NaNs that might have slipped through (shouldn't if preprocessing is robust)
        features = torch.nan_to_num(features, nan=0.0)
        return features

    def _get_labels(self, idx: int) -> torch.Tensor:
        """Get the multi-hot encoded labels for the given index."""
        return torch.tensor(self.encoded_labels[idx], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __getitem__")


class EmbeddedChestXrayDataset(ChestXrayDataset):
    """Dataset that embeds tabular data into the image as visual elements."""

    def __init__(
        self,
        image_dir: Path,
        metadata_path: Path,
        split_file_path: Path,
        split: Literal["train", "val", "test"],
        box_size: int = 20,
        margin: int = 2,
        position: Literal["top", "bottom", "right"] = "right",
        transform=None,
        target_transform=None,
        mlb: Optional[MultiLabelBinarizer] = None,
    ):
        super().__init__(
            image_dir,
            metadata_path,
            split_file_path,
            split,
            transform,
            target_transform,
            mlb,
        )
        self.box_size = box_size
        self.margin = margin
        self.position = position

    def _embed_tabular_data(  # Note: This function needs features before normalization
        self, image: Image.Image, raw_features: torch.Tensor
    ) -> Image.Image:
        """Embed raw (non-normalized) tabular features as visual elements.

        Args:
            image: Original image
            raw_features: Tensor of [age, gender_encoded, view_position_encoded]

        Returns:
            Modified image with embedded tabular data.
        """
        # Age: Grayscale intensity (clamp age for visualization)
        age = np.clip(raw_features[0].item(), 0, 100)
        age_intensity = int((age / 100.0) * 255)
        age_color = (age_intensity, age_intensity, age_intensity)

        # Gender: Binary (0=M=Black, 1=F=White)
        gender_color = (255, 255, 255) if raw_features[1].item() == 1.0 else (0, 0, 0)

        # View Position: Binary (0=PA=Black, 1=AP=White)
        view_color = (255, 255, 255) if raw_features[2].item() == 1.0 else (0, 0, 0)

        colors = [age_color, gender_color, view_color]

        # Create new image with space for metadata
        if self.position in ["top", "bottom"]:
            new_width = image.width
            new_height = image.height + self.box_size + 2 * self.margin
            box_y = (
                self.margin if self.position == "top" else image.height + self.margin
            )
            paste_y = 0 if self.position == "top" else self.box_size + 2 * self.margin
            new_image = Image.new(
                "RGB", (new_width, new_height), color="gray"
            )  # Use gray background
            new_image.paste(image, (0, paste_y))
            for i, color in enumerate(colors):
                box_x = self.margin + i * (self.box_size + self.margin)
                img_draw = Image.new("RGB", (self.box_size, self.box_size), color)
                new_image.paste(img_draw, (box_x, box_y))
        else:  # right
            new_width = image.width + self.box_size + 2 * self.margin
            new_height = image.height
            box_x = image.width + self.margin
            new_image = Image.new(
                "RGB", (new_width, new_height), color="gray"
            )  # Use gray background
            new_image.paste(image, (0, 0))
            for i, color in enumerate(colors):
                box_y = self.margin + i * (self.box_size + self.margin)
                img_draw = Image.new("RGB", (self.box_size, self.box_size), color)
                new_image.paste(img_draw, (box_x, box_y))

        return new_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(idx)
        # Get raw features *before* normalization for embedding visualization
        raw_features = self._get_tabular_features(idx)

        # Embed raw tabular data into image
        image = self._embed_tabular_data(image, raw_features)

        # Apply transforms (which include normalization) to the combined image
        if self.transform:
            image = self.transform(image)

        # Get labels
        labels = self._get_labels(idx)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels


class FusionChestXrayDataset(ChestXrayDataset):
    """Dataset that returns image and tabular data separately for fusion in the model."""

    def __init__(
        self,
        image_dir: Path,
        metadata_path: Path,
        split_file_path: Path,
        split: Literal["train", "val", "test"],
        transform=None,
        target_transform=None,
        mlb: Optional[MultiLabelBinarizer] = None,
        # Add normalizer for consistent tabular feature scaling
        tabular_normalizer: Optional[Any] = None,
    ):
        super().__init__(
            image_dir,
            metadata_path,
            split_file_path,
            split,
            transform,
            target_transform,
            mlb,
        )
        self.tabular_normalizer = tabular_normalizer
        if self.tabular_normalizer is None and split != "train":
            raise ValueError("Tabular normalizer must be provided for val/test splits.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._load_image(idx)
        # Get raw features
        features = self._get_tabular_features(idx)

        # Apply image transforms
        if self.transform:
            image = self.transform(image)

        # Normalize tabular features using the provided normalizer
        if self.tabular_normalizer:
            # Reshape features to [1, num_features] for normalizer
            features = features.unsqueeze(0)
            features = self.tabular_normalizer.transform(features)
            features = features.squeeze(0)  # Reshape back

        # Get labels
        labels = self._get_labels(idx)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, features, labels
