"""Unit tests for the dataset module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.data.dataset import ChestXrayDataset
from src.data.download import FilePaths


class TestChestXrayDataset(unittest.TestCase):
    """Unit tests for the ChestXrayDataset class."""

    def setUp(self):
        """Set up test data and environment."""
        # Create a temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.images_dir = self.temp_dir / "images"
        self.images_dir.mkdir()

        # Create test images (10 images for better randomization testing)
        self.image_names = [f"{str(i).zfill(8)}_000.png" for i in range(1, 11)]
        for img_name in self.image_names:
            # Create a simple test image (3 channels, 32x32)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(self.images_dir / img_name)

        # Create test clinical data
        self.clinical_data = pd.DataFrame(
            {
                "Image Index": self.image_names,
                "Finding Labels": [
                    "Cardiomegaly",
                    "No Finding",
                    "Edema",
                    "Mass",
                    "Nodule",
                    "Pneumonia",
                    "Atelectasis",
                    "Effusion",
                    "Infiltration",
                    "Pneumothorax",
                ],
                "Follow-up #": list(range(10)),
                "Patient Age": [
                    "058Y",
                    "012M",
                    "045Y",
                    "067Y",
                    "023Y",
                    "034Y",
                    "078Y",
                    "019Y",
                    "056Y",
                    "042Y",
                ],
                "Patient Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
                "View Position": [
                    "PA",
                    "AP",
                    "PA",
                    "AP",
                    "PA",
                    "AP",
                    "PA",
                    "AP",
                    "PA",
                    "AP",
                ],
            }
        )

        # Save clinical data to a temporary CSV
        self.clinical_data_path = self.temp_dir / "clinical_data.csv"
        self.clinical_data.to_csv(self.clinical_data_path, index=False)

        # Create FilePaths object
        self.file_paths = FilePaths(
            images_dir=Path(self.images_dir),
            clinical_data=Path(self.clinical_data_path),
        )

    def test_initialization(self):
        """Test dataset initialization."""
        dataset = ChestXrayDataset(self.file_paths)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.mode, "image_only")
        self.assertIsInstance(dataset.transform, transforms.ToTensor)

    def test_invalid_mode(self):
        """Test initialization with invalid mode."""
        with self.assertRaises(ValueError):
            ChestXrayDataset(self.file_paths, mode="invalid_mode")

    def test_image_only_mode(self):
        """Test dataset in image_only mode."""
        dataset = ChestXrayDataset(self.file_paths, mode="image_only")
        image, labels = dataset[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(labels.shape[0], 15)  # 15 possible conditions

    def test_image_and_tabular_mode(self):
        """Test dataset in image_and_tabular mode."""
        dataset = ChestXrayDataset(self.file_paths, mode="image_and_tabular")
        image, tabular, labels = dataset[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertIsInstance(tabular, torch.Tensor)
        self.assertEqual(tabular.shape[0], 4)  # 4 tabular features
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(labels.shape[0], 15)

    def test_embedded_image_mode(self):
        """Test dataset in embedded_image mode."""
        dataset = ChestXrayDataset(self.file_paths, mode="embedded_image")
        image, labels = dataset[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(labels.shape[0], 15)

    def test_transform_application(self):
        """Test that transforms are correctly applied."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])]
        )
        dataset = ChestXrayDataset(self.file_paths, transform=transform)
        image, _ = dataset[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 32, 32))

    def test_different_seeds(self):
        """Test that different seeds produce different data orderings."""
        dataset1 = ChestXrayDataset(self.file_paths, seed=42)
        dataset2 = ChestXrayDataset(self.file_paths, seed=42069)

        # Compare the actual data ordering using image names
        order1 = dataset1.tabular_df["imageIndex"].tolist()
        order2 = dataset2.tabular_df["imageIndex"].tolist()

        # With 10 items, the probability of getting same order is 1 in 3,628,800 (10!)
        self.assertNotEqual(
            order1, order2, "Different seeds should produce different image orderings"
        )

    def test_same_seed_reproducibility(self):
        """Test that same seed produces same data ordering."""
        dataset1 = ChestXrayDataset(self.file_paths, seed=42)
        dataset2 = ChestXrayDataset(self.file_paths, seed=42)

        # Compare the actual data ordering using image names
        order1 = dataset1.tabular_df["imageIndex"].tolist()
        order2 = dataset2.tabular_df["imageIndex"].tolist()

        self.assertEqual(order1, order2)

    def tearDown(self):
        """Clean up temporary files and directories."""
        import shutil

        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
