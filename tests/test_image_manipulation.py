"""Unit tests for image manipulation functions."""

import unittest

import torch

from src.utils.image_manipulation import embed_clinical_data_into_image, pad_image


class TestPadImage(unittest.TestCase):
    """Unit tests for the pad_image function."""

    def setUp(self):
        """Create common test data."""
        self.image = torch.zeros(3, 32, 32)

    def test_basic_padding(self):
        """Test basic padding functionality with default padding size."""
        result = pad_image(self.image)
        expected_shape = (3, 64, 64)  # 32 + 16 + 16 = 64 for both dimensions
        self.assertEqual(result.shape, expected_shape)

    def test_custom_padding(self):
        """Test padding with custom padding size."""
        padding = 8
        result = pad_image(self.image, padding=padding)
        expected_shape = (3, 48, 48)  # 32 + 8 + 8 = 48 for both dimensions
        self.assertEqual(result.shape, expected_shape)

    def test_invalid_input_shape(self):
        """Test handling of invalid input tensor shapes."""
        # Test 2D tensor
        with self.assertRaises(ValueError):
            pad_image(torch.zeros(32, 32))

    def test_batch_padding(self):
        """Test padding functionality with batched input."""
        # Create a batch of 4 images
        batch_image = torch.zeros(4, 3, 32, 32)
        result = pad_image(batch_image)
        expected_shape = (4, 3, 64, 64)  # 32 + 16 + 16 = 64 for both dimensions
        self.assertEqual(result.shape, expected_shape)

    def test_batch_padding_values(self):
        """Test that padding values are zero for batched input."""
        batch_size = 3
        batch_image = torch.ones(batch_size, 3, 32, 32)  # All ones
        result = pad_image(batch_image)

        # Check corners (should be part of padding) for all images in batch
        for i in range(batch_size):
            self.assertTrue(torch.allclose(result[i, :, :16, :16], torch.tensor(0.0)))
            self.assertTrue(torch.allclose(result[i, :, :16, -16:], torch.tensor(0.0)))
            self.assertTrue(torch.allclose(result[i, :, -16:, :16], torch.tensor(0.0)))
            self.assertTrue(torch.allclose(result[i, :, -16:, -16:], torch.tensor(0.0)))

            # Check that center content is preserved (all ones)
            self.assertTrue(
                torch.allclose(
                    result[i, :, 16:48, 16:48],
                    torch.ones_like(result[i, :, 16:48, 16:48]),
                )
            )

    def test_various_batch_sizes(self):
        """Test padding with different batch sizes."""
        batch_sizes = [1, 2, 8, 16]
        for size in batch_sizes:
            batch_image = torch.zeros(size, 3, 32, 32)
            result = pad_image(batch_image)
            expected_shape = (size, 3, 64, 64)
            self.assertEqual(result.shape, expected_shape)

    def test_padding_values(self):
        """Test that padding values are zero."""
        result = pad_image(self.image)
        # Check corners (should be part of padding)
        self.assertTrue(torch.allclose(result[:, :16, :16], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(result[:, :16, -16:], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(result[:, -16:, :16], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(result[:, -16:, -16:], torch.tensor(0.0)))


class TestEmbedClinicalData(unittest.TestCase):
    """Unit tests for the embed_clinical_data_into_image function."""

    def setUp(self):
        """Create common test data."""
        self.single_image = torch.zeros(1, 3, 32, 32)
        self.batch_image = torch.zeros(4, 3, 32, 32)

    def test_basic_functionality(self):
        """Test basic embedding with valid inputs."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])  # Single sample
        result = embed_clinical_data_into_image(self.single_image, tabular_data)

        # Check shape preservation
        self.assertEqual(result.shape, self.single_image.shape)

        # Check embedded values in each quadrant
        quad_size = 8  # matrix_size=16 // 2
        self.assertTrue(
            torch.allclose(
                result[0, 0, :quad_size, :quad_size], torch.tensor(0.5)
            )  # Follow-up
        )
        self.assertTrue(
            torch.allclose(
                result[0, 0, :quad_size, quad_size:16], torch.tensor(0.5)
            )  # Age
        )
        self.assertTrue(
            torch.allclose(
                result[0, 0, quad_size:16, :quad_size], torch.tensor(0.0)
            )  # Gender
        )
        self.assertTrue(
            torch.allclose(
                result[0, 0, quad_size:16, quad_size:16], torch.tensor(0.0)
            )  # Position
        )

    def test_batch_functionality(self):
        """Test batch embedding with valid inputs."""
        tabular_data = torch.tensor(
            [
                [0.1, 0.2, 0, 0],  # Sample 1
                [0.3, 0.4, 1, 0],  # Sample 2
                [0.5, 0.6, 0, 1],  # Sample 3
                [0.7, 0.8, 1, 1],  # Sample 4
            ]
        )
        result = embed_clinical_data_into_image(self.batch_image, tabular_data)

        # Check shape preservation
        self.assertEqual(result.shape, self.batch_image.shape)

        # Check values for first image in batch
        quad_size = 8
        self.assertTrue(
            torch.allclose(result[0, 0, :quad_size, :quad_size], torch.tensor(0.1))
        )
        self.assertTrue(
            torch.allclose(result[0, 0, :quad_size, quad_size:16], torch.tensor(0.2))
        )
        self.assertTrue(
            torch.allclose(result[0, 0, quad_size:16, :quad_size], torch.tensor(0.0))
        )
        self.assertTrue(
            torch.allclose(result[0, 0, quad_size:16, quad_size:16], torch.tensor(0.0))
        )

    def test_batch_size_mismatch(self):
        """Test validation of batch sizes between image and tabular data."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0], [0.3, 0.3, 1, 1]])  # 2 samples
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(self.batch_image, tabular_data)  # 4 images

    def test_invalid_gender_values(self):
        """Test validation of gender values (must be 0 or 1)."""
        tabular_data = torch.tensor([[0.5, 0.5, 0.5, 0]])  # Invalid gender
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(self.single_image, tabular_data)

    def test_invalid_position_values(self):
        """Test validation of position values (must be 0 or 1)."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0.5]])  # Invalid position
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(self.single_image, tabular_data)

    def test_image_modification_in_place(self):
        """Test that the original image is modified in place."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])
        original = self.single_image.clone()
        _ = embed_clinical_data_into_image(self.single_image, tabular_data)
        self.assertFalse(torch.equal(self.single_image, original))

    def test_matrix_size_validation(self):
        """Test matrix size validation."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])

        # Test invalid sizes
        invalid_sizes = [-1, 0, 15]  # negative, zero, odd
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                embed_clinical_data_into_image(
                    self.single_image, tabular_data, matrix_size=size
                )

    def test_image_size_validation(self):
        """Test image size validation against matrix size."""
        small_image = torch.zeros(1, 3, 8, 8)
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(small_image, tabular_data, matrix_size=16)

    def test_all_channels_same(self):
        """Test that clinical data is embedded identically in all channels."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])
        result = embed_clinical_data_into_image(self.single_image, tabular_data)

        # Compare all channels to first channel
        for c in range(1, result.shape[1]):
            self.assertTrue(torch.equal(result[:, 0, :16, :16], result[:, c, :16, :16]))


if __name__ == "__main__":
    unittest.main()
