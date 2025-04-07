"""Unit tests for preprocessing functions."""

import unittest
import shutil
import tempfile
import pandas as pd
import torch
from src.utils.preprocessing import (
    generate_image_labels,
    convert_agestr_to_years,
    create_working_tabular_df,
)


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
            labels = generate_image_labels(condition)
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
        labels = generate_image_labels(input_str)

        self.assertEqual(labels[0], 1)  # Atelectasis
        self.assertEqual(labels[3], 1)  # Edema
        self.assertEqual(labels[9], 1)  # Mass
        self.assertEqual(sum(labels), 3)  # Only three positions should be 1

    def test_case_insensitivity(self):
        """Test that the function is case insensitive."""
        variations = ["ATELECTASIS", "atelectasis", "Atelectasis", "aTeLeCtAsIs"]

        for variant in variations:
            labels = generate_image_labels(variant)
            self.assertEqual(labels[0], 1)
            self.assertEqual(sum(labels), 1)

    def test_no_finding(self):
        """Test the 'No Finding' case."""
        labels = generate_image_labels("No Finding")
        self.assertEqual(labels[10], 1)
        self.assertEqual(sum(labels), 1)

    def test_empty_string(self):
        """Test empty string input."""
        with self.assertRaises(ValueError):
            generate_image_labels("")

    def test_invalid_label(self):
        """Test with invalid/non-existent condition."""
        with self.assertRaises(ValueError):
            generate_image_labels("NonExistentCondition")

    def test_mixed_valid_invalid(self):
        """Test mixture of valid and invalid labels."""
        with self.assertRaises(ValueError):
            generate_image_labels("Atelectasis|NonExistentCondition")

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
        labels = generate_image_labels(input_str)
        self.assertEqual(sum(labels), 14)  # All labels except "No Finding" should be 1
        self.assertEqual(labels[10], 0)

    def test_shape(self):
        """Test the shape of the output tensor."""
        input_str = "Atelectasis|Edema|Mass"
        labels = generate_image_labels(input_str)
        self.assertEqual(labels.shape, (15,))
        self.assertEqual(labels.dtype, torch.float32)


class TestAgeConversion(unittest.TestCase):
    """Unit tests for the age string conversion function."""

    def test_years(self):
        """Test conversion of year-based age strings."""
        self.assertEqual(convert_agestr_to_years("045y"), 45.0)
        self.assertEqual(convert_agestr_to_years("001y"), 1.0)
        self.assertEqual(convert_agestr_to_years("000y"), 0.0)

    def test_months(self):
        """Test conversion of month-based age strings."""
        self.assertEqual(convert_agestr_to_years("012m"), 1.0)
        self.assertEqual(convert_agestr_to_years("006m"), 0.5)
        self.assertEqual(convert_agestr_to_years("024m"), 2.0)

    def test_days(self):
        """Test conversion of day-based age strings."""
        self.assertAlmostEqual(convert_agestr_to_years("365d"), 1.0, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("180d"), 0.493151, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("030d"), 0.082192, places=6)

    def test_weeks(self):
        """Test conversion of week-based age strings."""
        self.assertAlmostEqual(convert_agestr_to_years("052w"), 1.0, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("026w"), 0.5, places=6)
        self.assertAlmostEqual(convert_agestr_to_years("013w"), 0.25, places=6)

    def test_case_sensitivity(self):
        """Test that the function handles different cases properly."""
        self.assertEqual(convert_agestr_to_years("045Y"), 45.0)
        self.assertEqual(convert_agestr_to_years("012M"), 1.0)
        self.assertAlmostEqual(convert_agestr_to_years("365D"), 1.0, places=6)
        self.assertEqual(convert_agestr_to_years("052W"), 1.0)

    def test_empty_string(self):
        """Test that empty strings raise ValueError."""
        with self.assertRaises(ValueError):
            convert_agestr_to_years("")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("    ")

    def test_invalid_length(self):
        """Test that strings of invalid length raise ValueError."""
        with self.assertRaises(ValueError):
            convert_agestr_to_years("1y")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("45yr")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("1000y")

    def test_invalid_format(self):
        """Test that strings with invalid format raise ValueError."""
        with self.assertRaises(ValueError):
            convert_agestr_to_years("045x")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("abcy")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("45.y")
        with self.assertRaises(ValueError):
            convert_agestr_to_years("-45y")


class TestTabularDataPreprocessing(unittest.TestCase):
    """Unit tests for the tabular data preprocessing function."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame(
            {
                "Image Index": ["00000001_000.png", "00000002_000.png"],
                "Finding Labels": ["Cardiomegaly", "No Finding"],
                "Follow-up #": [0, 1],
                "Patient Age": ["058Y", "012M"],
                "Patient Gender": ["M", "F"],
                "View Position": ["PA", "AP"],
            }
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_column_renaming(self):
        """Test that columns are correctly renamed."""
        result_df = create_working_tabular_df(self.test_df)
        expected_columns = {
            "imageIndex",
            "followUpNumber",
            "patientAge",
            "patientGender",
            "viewPosition",
        }
        self.assertTrue(expected_columns.issubset(set(result_df.columns)))

    def test_age_conversion(self):
        """Test age string conversion."""
        result_df = create_working_tabular_df(self.test_df)
        self.assertEqual(result_df["patientAge"].iloc[0], 58.0)  # 058Y
        self.assertEqual(result_df["patientAge"].iloc[1], 1.0)  # 012M

    def test_gender_encoding(self):
        """Test gender binary encoding."""
        result_df = create_working_tabular_df(self.test_df)
        self.assertEqual(result_df["patientGender"].iloc[0], 0)  # M
        self.assertEqual(result_df["patientGender"].iloc[1], 1)  # F

    def test_view_position_encoding(self):
        """Test view position binary encoding."""
        result_df = create_working_tabular_df(self.test_df)
        self.assertEqual(result_df["viewPosition"].iloc[0], 0)  # PA
        self.assertEqual(result_df["viewPosition"].iloc[1], 1)  # AP

    def test_label_encoding(self):
        """Test one-hot encoding of finding labels."""
        result_df = create_working_tabular_df(self.test_df)

        # Test Cardiomegaly encoding
        self.assertEqual(result_df["label_cardiomegaly"].iloc[0], 1)
        self.assertEqual(result_df["label_no_finding"].iloc[0], 0)

        # Test No Finding encoding
        self.assertEqual(result_df["label_cardiomegaly"].iloc[1], 0)
        self.assertEqual(result_df["label_no_finding"].iloc[1], 1)

    def test_case_insensitivity(self):
        """Test case insensitive handling of categorical variables."""
        test_df = self.test_df.copy()
        test_df["Patient Gender"] = ["M", "f"]
        test_df["View Position"] = ["pA", "Ap"]

        result_df = create_working_tabular_df(test_df)
        self.assertEqual(result_df["patientGender"].iloc[0], 0)  # m
        self.assertEqual(result_df["patientGender"].iloc[1], 1)  # f
        self.assertEqual(result_df["viewPosition"].iloc[0], 0)  # pa
        self.assertEqual(result_df["viewPosition"].iloc[1], 1)  # ap


if __name__ == "__main__":
    unittest.main()
