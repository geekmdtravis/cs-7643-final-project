"""Unit tests for the dataloaders module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.create_dataloaders import create_dataloaders
from src.data.download import FilePaths


class TestCreateDataloaders(unittest.TestCase):
    """Unit tests for the create_dataloaders function."""

    def setUp(self):
        """Set up test data and environment."""
        # Create a temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.images_dir = self.temp_dir / "images"
        self.images_dir.mkdir()

        # Create test images
        self.image_names = [f"image_{i}.png" for i in range(10)]  # 10 test images
        for img_name in self.image_names:
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(self.images_dir / img_name)

        # Create test clinical data
        self.clinical_data = pd.DataFrame(
            {
                "Image Index": self.image_names,
                "Finding Labels": ["No Finding"] * 10,
                "Follow-up #": list(range(10)),
                "Patient Age": ["058Y"] * 10,
                "Patient Gender": ["M", "F"] * 5,
                "View Position": ["PA", "AP"] * 5,
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

    def test_basic_functionality(self):
        """Test basic functionality of create_dataloaders."""
        train_loader, test_loader = create_dataloaders(self.file_paths)

        # Check that we get DataLoader instances
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        # Check batch size
        self.assertEqual(train_loader.batch_size, 32)
        self.assertEqual(test_loader.batch_size, 32)

    def test_train_test_split(self):
        """Test train/test split ratio."""
        train_ratio = 0.8
        train_loader, test_loader = create_dataloaders(
            self.file_paths, train_ratio=train_ratio
        )

        # Calculate expected sizes
        total_samples = len(self.image_names)
        expected_train_size = int(total_samples * train_ratio)
        expected_test_size = total_samples - expected_train_size

        # Get actual sizes from dataloaders
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)

        self.assertEqual(train_size, expected_train_size)
        self.assertEqual(test_size, expected_test_size)

    def test_batch_size_parameter(self):
        """Test custom batch size."""
        batch_size = 4
        train_loader, test_loader = create_dataloaders(
            self.file_paths, batch_size=batch_size
        )

        self.assertEqual(train_loader.batch_size, batch_size)
        self.assertEqual(test_loader.batch_size, batch_size)

    def test_seed_reproducibility(self):
        """Test that same seed produces same split."""
        seed = 42
        train_loader1, test_loader1 = create_dataloaders(self.file_paths, seed=seed)
        train_loader2, test_loader2 = create_dataloaders(self.file_paths, seed=seed)

        # Compare indices of datasets
        train_indices1 = train_loader1.dataset.indices
        train_indices2 = train_loader2.dataset.indices
        test_indices1 = test_loader1.dataset.indices
        test_indices2 = test_loader2.dataset.indices

        self.assertEqual(train_indices1, train_indices2)
        self.assertEqual(test_indices1, test_indices2)

    def test_num_workers(self):
        """Test different numbers of workers."""
        num_workers = 2
        train_loader, test_loader = create_dataloaders(
            self.file_paths, num_workers=num_workers
        )

        self.assertEqual(train_loader.num_workers, num_workers)
        self.assertEqual(test_loader.num_workers, num_workers)

    def test_invalid_train_ratio(self):
        """Test invalid train_ratio values."""
        invalid_ratios = [-0.1, 0, 1.0, 1.1]
        for ratio in invalid_ratios:
            with self.assertRaises(ValueError):
                create_dataloaders(self.file_paths, train_ratio=ratio)

    def test_invalid_batch_size(self):
        """Test invalid batch_size values."""
        invalid_batch_sizes = [0, -1]
        for batch_size in invalid_batch_sizes:
            with self.assertRaises(ValueError):
                create_dataloaders(self.file_paths, batch_size=batch_size)

    def test_invalid_num_workers(self):
        """Test invalid num_workers values."""
        invalid_num_workers = [-1]
        for num_workers in invalid_num_workers:
            with self.assertRaises(ValueError):
                create_dataloaders(self.file_paths, num_workers=num_workers)

    def test_invalid_seed(self):
        """Test invalid seed values."""
        invalid_seeds = [-1]
        for seed in invalid_seeds:
            with self.assertRaises(ValueError):
                create_dataloaders(self.file_paths, seed=seed)

    def test_invalid_file_paths(self):
        """Test invalid file_paths values."""
        invalid_file_paths = [None, "invalid_path"]
        for file_path in invalid_file_paths:
            with self.assertRaises(ValueError):
                create_dataloaders(file_path)

    def test_data_shapes_and_values(self):
        """Test shapes and values of data from dataloaders."""
        # Test with different batch sizes
        for batch_size in [1, 3, 5]:  # Test edge cases and max size for 0.5 split
            train_loader, test_loader = create_dataloaders(
                self.file_paths, batch_size=batch_size, train_ratio=0.5
            )

            # Test both train and test loaders
            for loader in [train_loader, test_loader]:
                batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = next(
                    iter(loader)
                )
                images, tabular, labels = batch

                # Image tensor tests
                self.assertEqual(len(images.shape), 4)  # B x C x H x W
                self.assertEqual(images.shape[0], batch_size)  # Batch size
                self.assertEqual(images.shape[1], 3)  # RGB channels
                self.assertEqual(images.shape[2], 32)  # Height
                self.assertEqual(images.shape[3], 32)  # Width
                self.assertTrue(images.is_contiguous())  # Memory layout

                # Test image normalization (using ImageNet stats)
                self.assertTrue(
                    torch.all((images[:, 0] >= -2.5) & (images[:, 0] <= 2.5))
                )  # Reasonable range after normalization

                # Tabular features tests
                self.assertEqual(len(tabular.shape), 2)  # B x F
                self.assertEqual(tabular.shape[0], batch_size)  # Batch size
                self.assertEqual(tabular.shape[1], 4)  # 4 features
                self.assertTrue(tabular.is_contiguous())
                self.assertEqual(tabular.dtype, torch.float32)

                # Verify tabular value ranges
                age_values = tabular[:, 0]
                gender_values = tabular[:, 1]
                view_position_values = tabular[:, 2]
                followup_values = tabular[:, 3]

                self.assertTrue(torch.all((age_values == 58)))
                self.assertTrue(torch.all((gender_values == 0) | (gender_values == 1)))
                self.assertTrue(
                    torch.all((view_position_values == 0) | (view_position_values == 1))
                )
                self.assertTrue(
                    torch.all(followup_values >= 0)
                )  # Non-negative followup numbers

                # Labels tests
                self.assertEqual(len(labels.shape), 2)  # B x Classes
                self.assertEqual(labels.shape[0], batch_size)
                self.assertEqual(labels.shape[1], 15)  # 15 conditions
                self.assertTrue(labels.is_contiguous())
                self.assertEqual(labels.dtype, torch.float32)

                # Verify labels are binary
                self.assertTrue(torch.all((labels == 0) | (labels == 1)))

                # Test pin_memory is set
                self.assertTrue(loader.pin_memory)

                # Verify batch alignment
                self.assertEqual(images.shape[0], tabular.shape[0])
                self.assertEqual(images.shape[0], labels.shape[0])

    def tearDown(self):
        """Clean up temporary files and directories."""
        import shutil

        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
