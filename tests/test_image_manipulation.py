"""Unit tests for image manipulation functions."""

import unittest

import torch

from src.utils.image_manipulation import (
    embed_clinical_data_into_image,
    embed_clinical_data_into_image_alt,
    pad_image,
)


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


class TestEmbedClinicalDataAlt(unittest.TestCase):
    """Unit tests for the embed_clinical_data_into_image_alt function."""

    def setUp(self):
        """Create common test data."""
        self.single_image = torch.zeros(1, 3, 32, 32)
        self.batch_image = torch.zeros(4, 3, 32, 32)

    def test_basic_functionality(self):
        """Test basic embedding with valid inputs."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])  # Single sample
        result = embed_clinical_data_into_image_alt(self.single_image, tabular_data)

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
        result = embed_clinical_data_into_image_alt(self.batch_image, tabular_data)

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
            embed_clinical_data_into_image_alt(
                self.batch_image, tabular_data
            )  # 4 images

    def test_invalid_gender_values(self):
        """Test validation of gender values (must be 0 or 1)."""
        tabular_data = torch.tensor([[0.5, 0.5, 0.5, 0]])  # Invalid gender
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image_alt(self.single_image, tabular_data)

    def test_invalid_position_values(self):
        """Test validation of position values (must be 0 or 1)."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0.5]])  # Invalid position
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image_alt(self.single_image, tabular_data)

    def test_image_modificatio_in_place(self):
        """Test that the original image is modified in place."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])
        original = self.single_image.clone()
        _ = embed_clinical_data_into_image_alt(self.single_image, tabular_data)
        self.assertFalse(torch.equal(self.single_image, original))

    def test_matrix_size_validation(self):
        """Test matrix size validation."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])

        # Test invalid sizes
        invalid_sizes = [-1, 0, 15]  # negative, zero, odd
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                embed_clinical_data_into_image_alt(
                    self.single_image, tabular_data, matrix_size=size
                )

    def test_image_size_validation(self):
        """Test image size validation against matrix size."""
        small_image = torch.zeros(1, 3, 8, 8)
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image_alt(
                small_image, tabular_data, matrix_size=16
            )

    def test_all_channels_same(self):
        """Test that clinical data is embedded identically in all channels."""
        tabular_data = torch.tensor([[0.5, 0.5, 0, 0]])
        result = embed_clinical_data_into_image_alt(self.single_image, tabular_data)

        # Compare all channels to first channel
        for c in range(1, result.shape[1]):
            self.assertTrue(torch.equal(result[:, 0, :16, :16], result[:, c, :16, :16]))


class TestEmbedClinicalData(unittest.TestCase):
    """Unit tests for the embed_clinical_data_into_image function."""

    def setUp(self):
        """Create common test data."""
        self.image = torch.zeros(3, 32, 32)

    def test_basic_functionality(self):
        """Test basic embedding with valid inputs."""
        result = embed_clinical_data_into_image(
            image=self.image,
            age=60,
            gender="female",
            xr_pos="PA",
            xr_count=5,
            matrix_size=16,
        )

        # Check shape preservation
        self.assertEqual(result.shape, self.image.shape)

        # Check embedded values in each quadrant
        self.assertTrue(
            torch.allclose(result[:, 0:8, 0:8], torch.tensor(60 / 120))
        )  # Age
        self.assertTrue(
            torch.allclose(result[:, 0:8, 8:16], torch.tensor(1.0))
        )  # Gender (female)
        self.assertTrue(
            torch.allclose(result[:, 8:16, 0:8], torch.tensor(1.0))
        )  # XR pos (PA)
        self.assertTrue(
            torch.allclose(result[:, 8:16, 8:16], torch.tensor(5 / 10))
        )  # XR count

    def test_image_cloning(self):
        """Test that the original image is not modified."""
        original = self.image.clone()
        _ = embed_clinical_data_into_image(
            image=self.image, age=60, gender="male", xr_pos="AP", xr_count=5
        )
        self.assertTrue(torch.equal(self.image, original))

    def test_invalid_image_shape(self):
        """Test handling of invalid image shapes."""
        # Test 2D tensor
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=torch.zeros(32, 32),
                age=60,
                gender="male",
                xr_pos="AP",
                xr_count=5,
            )

    def test_padding_size_validation(self):
        """Test padding size validation."""
        invalid_sizes = [-1, 0, 15]  # negative, zero, odd
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                embed_clinical_data_into_image(
                    image=self.image,
                    age=60,
                    gender="male",
                    xr_pos="AP",
                    xr_count=5,
                    matrix_size=size,
                )

    def test_image_size_validation(self):
        """Test image size validation against padding size."""
        small_image = torch.zeros(3, 8, 8)
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=small_image,
                age=60,
                gender="male",
                xr_pos="AP",
                xr_count=5,
                matrix_size=16,
            )

    def test_age_validation(self):
        """Test age validation."""
        invalid_ages = [-1, 121]
        for age in invalid_ages:
            with self.assertRaises(ValueError):
                embed_clinical_data_into_image(
                    image=self.image, age=age, gender="male", xr_pos="AP", xr_count=5
                )

    def test_xr_count_validation(self):
        """Test X-ray count validation."""
        invalid_counts = [-1, -2]
        for count in invalid_counts:
            with self.assertRaises(ValueError):
                embed_clinical_data_into_image(
                    image=self.image, age=60, gender="male", xr_pos="AP", xr_count=count
                )

    def test_gender_validation(self):
        """Test gender validation."""
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=self.image,
                age=60,
                gender="invalid",  # Invalid gender value
                xr_pos="AP",
                xr_count=5,
            )

    def test_xr_pos_validation(self):
        """Test X-ray position validation."""
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=self.image,
                age=60,
                gender="male",
                xr_pos="invalid",  # Invalid position value
                xr_count=5,
            )

    def test_edge_cases(self):
        """Test edge cases for valid inputs."""
        # Test minimum values
        min_result = embed_clinical_data_into_image(
            image=self.image, age=1, gender="male", xr_pos="AP", xr_count=1
        )
        self.assertTrue(
            torch.allclose(min_result[:, 0:8, 0:8], torch.tensor(0.00833333))
        )  # Age
        self.assertTrue(
            torch.allclose(min_result[:, 0:8, 8:16], torch.tensor(0.0))
        )  # Gender
        self.assertTrue(
            torch.allclose(min_result[:, 8:16, 0:8], torch.tensor(0.0))
        )  # XR pos
        self.assertTrue(
            torch.allclose(min_result[:, 8:16, 8:16], torch.tensor(0.1))
        )  # XR count

        # Test maximum values
        max_result = embed_clinical_data_into_image(
            image=self.image, age=120, gender="female", xr_pos="PA", xr_count=10
        )
        self.assertTrue(
            torch.allclose(max_result[:, 0:8, 0:8], torch.tensor(1.0))
        )  # Age
        self.assertTrue(
            torch.allclose(max_result[:, 0:8, 8:16], torch.tensor(1.0))
        )  # Gender
        self.assertTrue(
            torch.allclose(max_result[:, 8:16, 0:8], torch.tensor(1.0))
        )  # XR pos
        self.assertTrue(
            torch.allclose(max_result[:, 8:16, 8:16], torch.tensor(1.0))
        )  # XR count

    def test_basic_batch_functionality(self):
        """Test basic batch embedding with valid inputs."""
        batch_size = 3
        batch_image = torch.zeros(batch_size, 3, 32, 32)
        ages = [30, 45, 60]
        genders = ["male", "female", "male"]
        positions = ["AP", "PA", "AP"]
        counts = [2, 3, 4]

        result = embed_clinical_data_into_image(
            image=batch_image,
            age=ages,
            gender=genders,
            xr_pos=positions,
            xr_count=counts,
            matrix_size=16,
        )

        # Check shape preservation
        self.assertEqual(result.shape, batch_image.shape)

        # Check embedded values for first image in batch
        self.assertTrue(
            torch.allclose(result[0, :, 0:8, 0:8], torch.tensor(30 / 120))
        )  # Age
        self.assertTrue(
            torch.allclose(result[0, :, 0:8, 8:16], torch.tensor(0.0))
        )  # Gender (male)
        self.assertTrue(
            torch.allclose(result[0, :, 8:16, 0:8], torch.tensor(0.0))
        )  # XR pos (AP)
        self.assertTrue(
            torch.allclose(result[0, :, 8:16, 8:16], torch.tensor(0.2))
        )  # XR count

    def test_batch_size_mismatch(self):
        """Test validation of batch sizes between image and clinical data."""
        batch_image = torch.zeros(3, 3, 32, 32)

        # Test mismatched age list
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=batch_image,
                age=[30, 45],  # Only 2 ages for 3 images
                gender=["male", "female", "male"],
                xr_pos=["AP", "PA", "AP"],
                xr_count=[2, 3, 4],
            )

    def test_batch_requires_lists(self):
        """Test that batch operations require lists for all clinical data."""
        batch_image = torch.zeros(2, 3, 32, 32)

        # Test with single value for age
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=batch_image,
                age=50,  # Single value instead of list
                gender=["male", "female"],
                xr_pos=["AP", "PA"],
                xr_count=[3, 4],
            )

        # Test with single value for gender
        with self.assertRaises(ValueError):
            embed_clinical_data_into_image(
                image=batch_image,
                age=[30, 40],
                gender="male",  # Single value instead of list
                xr_pos=["AP", "PA"],
                xr_count=[3, 4],
            )

    def test_batch_edge_cases(self):
        """Test edge cases with batched inputs."""
        # Test single-item batch (should work like non-batch)
        single_batch = torch.zeros(1, 3, 32, 32)
        result = embed_clinical_data_into_image(
            image=single_batch,
            age=[60],
            gender=["female"],
            xr_pos=["PA"],
            xr_count=[5],
        )
        self.assertEqual(result.shape, single_batch.shape)

        # Test large batch
        large_batch = torch.zeros(10, 3, 32, 32)
        ages = list(range(20, 120, 10))  # 10 ages
        genders = ["male", "female"] * 5  # 10 genders
        positions = ["AP", "PA"] * 5  # 10 positions
        counts = list(range(1, 11))  # 10 counts

        result = embed_clinical_data_into_image(
            image=large_batch,
            age=ages,
            gender=genders,
            xr_pos=positions,
            xr_count=counts,
        )
        self.assertEqual(result.shape, large_batch.shape)


if __name__ == "__main__":
    unittest.main()
