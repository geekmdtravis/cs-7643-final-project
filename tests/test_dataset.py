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


class TestChestXrayDataset(unittest.TestCase):
    """Test cases for the ChestXrayDataset class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.test_dir) / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.num_samples = 3
        self.image_size = (64, 64)
        self.image_names = []

        for i in range(self.num_samples):
            img_name = f"image_{i}.png"
            self.image_names.append(img_name)
            img = Image.fromarray(
                np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            )
            img.save(self.images_dir / img_name)

        self.clinical_data = pd.DataFrame(
            {
                "imageIndex": self.image_names,
                "patientAge": np.random.randint(20, 80, self.num_samples),
                "patientGender": np.random.randint(0, 2, self.num_samples),
                "viewPosition": np.random.randint(0, 2, self.num_samples),
                "followUpNumber": np.random.randint(0, 5, self.num_samples),
                "label_1": np.random.randint(0, 2, self.num_samples),
                "label_2": np.random.randint(0, 2, self.num_samples),
            }
        )

        self.clinical_data_path = Path(self.test_dir) / "clinical_data.csv"
        self.clinical_data.to_csv(self.clinical_data_path, index=False)

        self.dataset = ChestXrayDataset(
            clinical_data=self.clinical_data_path, cxr_images_dir=self.images_dir
        )

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil

        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test dataset initialization."""
        self.assertIsInstance(self.dataset, ChestXrayDataset)
        self.assertEqual(len(self.dataset.tabular_df), self.num_samples)

    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.dataset), self.num_samples)

    def test_getitem(self):
        """Test __getitem__ method."""
        idx = 0
        image, tabular_features, labels = self.dataset[idx]

        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(tabular_features, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

        self.assertEqual(image.shape, (3, *self.image_size))
        self.assertEqual(tabular_features.shape, (4,))
        self.assertEqual(labels.shape, (2,))

    def test_custom_transform(self):
        """Test dataset with custom transform."""
        custom_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32))]
        )

        dataset = ChestXrayDataset(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            transform=custom_transform,
        )

        image, _, _ = dataset[0]
        self.assertEqual(image.shape, (3, 32, 32))

    def test_invalid_index(self):
        """Test accessing invalid index."""
        with self.assertRaises(IndexError):
            _ = self.dataset[len(self.dataset)]


if __name__ == "__main__":
    unittest.main()
