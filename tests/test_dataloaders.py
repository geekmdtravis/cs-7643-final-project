"""Unit tests for the dataloaders module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.create_dataloader import create_dataloader


class TestDataloaders(unittest.TestCase):
    """Test cases for the dataloader creation functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.test_dir) / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.num_samples = 10
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

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil

        shutil.rmtree(self.test_dir)

    def test_create_dataloader(self):
        """Test basic dataloader creation."""
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=4,
            num_workers=0,
        )

        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, 4)

    def test_batch_generation(self):
        """Test if dataloader generates correct batches."""
        batch_size = 4
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=batch_size,
            num_workers=0,
        )

        images, tabular, labels = next(iter(loader))

        self.assertEqual(images.shape, (batch_size, 3, *self.image_size))
        self.assertEqual(tabular.shape, (batch_size, 4))
        self.assertEqual(labels.shape, (batch_size, 2))

    def test_invalid_batch_size(self):
        """Test if invalid batch size raises error."""
        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                batch_size=0,
            )

        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                batch_size=-1,
            )

    def test_invalid_num_workers(self):
        """Test if invalid num_workers raises error."""
        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                num_workers=-1,
            )

    def test_multi_worker_loading(self):
        """Test dataloader with multiple workers."""
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=4,
            num_workers=2,
        )

        for images, tabular, labels in loader:
            self.assertIsInstance(images, torch.Tensor)
            self.assertIsInstance(tabular, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            break


if __name__ == "__main__":
    unittest.main()
