"""Unit tests for preprocessing functions."""

import unittest
import torch
from src.utils.preprocessing import generate_image_lables


class TestPreprocessing(unittest.TestCase):
    """Unit tests for the preprocessing functions."""

    def test_single_labels(self):
        """Test detection of single labels."""
        conditions = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Hernia",
            "Infiltration",
            "Mass",
            "No Finding",
            "Nodule",
            "Pleural_Thickening",
            "Pneumonia",
            "Pneumothorax",
        ]

        for idx, condition in enumerate(conditions):
            labels = generate_image_lables(condition)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(labels.shape, (15,))
            self.assertEqual(labels.dtype, torch.float32)
            self.assertEqual(labels[idx], 1)
            # Verify other positions are 0
            zero_positions = list(range(15))
            zero_positions.remove(idx)
            for pos in zero_positions:
                self.assertEqual(labels[pos], 0)

    def test_multiple_labels(self):
        """Test detection of multiple conditions."""
        input_str = "Atelectasis|Edema|Mass"
        labels = generate_image_lables(input_str)

        self.assertEqual(labels[0], 1)  # Atelectasis
        self.assertEqual(labels[3], 1)  # Edema
        self.assertEqual(labels[9], 1)  # Mass
        self.assertEqual(sum(labels), 3)  # Only three positions should be 1

    def test_case_insensitivity(self):
        """Test that the function is case insensitive."""
        variations = ["ATELECTASIS", "atelectasis", "Atelectasis", "aTeLeCtAsIs"]

        for variant in variations:
            labels = generate_image_lables(variant)
            self.assertEqual(labels[0], 1)
            self.assertEqual(sum(labels), 1)

    def test_no_finding(self):
        """Test the 'No Finding' case."""
        labels = generate_image_lables("No Finding")
        self.assertEqual(labels[10], 1)
        self.assertEqual(sum(labels), 1)

    def test_empty_string(self):
        """Test empty string input."""
        with self.assertRaises(ValueError):
            generate_image_lables("")

    def test_invalid_label(self):
        """Test with invalid/non-existent condition."""
        with self.assertRaises(ValueError):
            generate_image_lables("NonExistentCondition")

    def test_mixed_valid_invalid(self):
        """Test mixture of valid and invalid labels."""
        with self.assertRaises(ValueError):
            generate_image_lables("Atelectasis|NonExistentCondition")

    def test_all_found_labels(self):
        """Test all labels are found."""
        input_str = "|".join(
            [
                "Atelectasis",
                "Cardiomegaly",
                "Consolidation",
                "Edema",
                "Effusion",
                "Emphysema",
                "Fibrosis",
                "Hernia",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pleural_Thickening",
                "Pneumonia",
                "Pneumothorax",
            ]
        )
        labels = generate_image_lables(input_str)
        self.assertEqual(sum(labels), 14)  # All labels except "No Finding" should be 1
        self.assertEqual(labels[10], 0)

    def test_shape(self):
        """Test the shape of the output tensor."""
        input_str = "Atelectasis|Edema|Mass"
        labels = generate_image_lables(input_str)
        self.assertEqual(labels.shape, (15,))
        self.assertEqual(labels.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
