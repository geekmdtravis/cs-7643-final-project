"""Preprocessing functions for medical image datasets."""

import random

import numpy as np
import pandas as pd
import torch


def generate_image_labels(finding_labels: str) -> torch.Tensor:
    """
    Generate image labels from finding labels.

    Args:
        finding_labels (str): A string of finding labels separated by '|'.

    Returns:
        torch.Tensor: A tensor representing the image labels of shape (15,).
    """
    _fl = finding_labels.lower()

    if _fl.strip() == "":
        raise ValueError("Finding labels cannot be an empty string.")

    valid_labels = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "effusion",
        "emphysema",
        "fibrosis",
        "hernia",
        "infiltration",
        "mass",
        "no finding",
        "nodule",
        "pleural_thickening",
        "pneumonia",
        "pneumothorax",
    ]

    for label in _fl.split("|"):
        if label not in valid_labels:
            raise ValueError(f"Invalid finding label: {label}")

    image_labels = torch.zeros(15, dtype=torch.float32)
    image_labels[0] = 1 if "atelectasis" in _fl else 0
    image_labels[1] = 1 if "cardiomegaly" in _fl else 0
    image_labels[2] = 1 if "consolidation" in _fl else 0
    image_labels[3] = 1 if "edema" in _fl else 0
    image_labels[4] = 1 if "effusion" in _fl else 0
    image_labels[5] = 1 if "emphysema" in _fl else 0
    image_labels[6] = 1 if "fibrosis" in _fl else 0
    image_labels[7] = 1 if "hernia" in _fl else 0
    image_labels[8] = 1 if "infiltration" in _fl else 0
    image_labels[9] = 1 if "mass" in _fl else 0
    image_labels[10] = 1 if "no finding" in _fl else 0
    image_labels[11] = 1 if "nodule" in _fl else 0
    image_labels[12] = 1 if "pleural_thickening" in _fl else 0
    image_labels[13] = 1 if "pneumonia" in _fl else 0
    image_labels[14] = 1 if "pneumothorax" in _fl else 0

    return image_labels


def convert_agestr_to_years(agestr: str) -> float:
    """
    Convert age string to years.

    Args:
        agestr (str): Age string in the format 'XXy' or 'XXm'.

    Returns:
        float: Age in years
    """
    _agestr = agestr.strip().lower()
    if not _agestr:
        raise ValueError("Age string cannot be empty.")
    if not (len(_agestr) == 4):
        raise ValueError(f"Invalid age string length: {agestr}")
    if not (_agestr[:-1].isdigit() and _agestr[-1] in ["y", "m", "d", "w"]):
        raise ValueError(f"Invalid age string format: {agestr}")

    age_value = float(_agestr[:-1])
    if _agestr.endswith("y"):
        return age_value
    elif _agestr.endswith("m"):
        return age_value / 12
    elif _agestr.endswith("d"):
        return age_value / 365
    elif _agestr.endswith("w"):
        return age_value / 52
    else:
        raise ValueError(f"Invalid age string format: {agestr}")


def create_working_tabular_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a working DataFrame for tabular data with standardized features. Of note,
    none of the preprocessing steps here will produce data leakage, as the
    transformations are applied element-wise and do not depend on the entire dataset.
    This function is designed to be used with the NIH Chest X-ray dataset.

    The function performs the following transformations:
    - Selects and renames relevant columns
    - Converts patient age from string to float (in years)
    - Converts patient gender to a binary 1/0 encoding (0=M, 1=F)
    - Converts view position to a binary 1/0 encoding (0=PA, 1=AP)
    - Generates one-hot encoded disease labels for 14 conditions and 1 "no finding"

    Args:
        df (pd.DataFrame): Input DataFrame containing medical image metadata
        from the NIH Chest X-ray dataset.

    Returns:
        pd.DataFrame: Processed DataFrame with the following columns:
            - imageIndex: Original image filename
            - followUpNumber: Patient follow-up visit number
            - patientAge: Age in years (float)
            - patientGender: Binary encoded gender (0=M, 1=F)
            - viewPosition: Binary encoded position (0=PA, 1=AP)
            - label_{condition}: One-hot encoded disease labels (15 columns)
    """
    # Select and rename relevant columns
    working_df = pd.DataFrame()
    working_df["imageIndex"] = df["Image Index"]
    working_df["followUpNumber"] = df["Follow-up #"]
    working_df["patientAge"] = df["Patient Age"].apply(convert_agestr_to_years)

    # Convert gender to binary (case-insensitive)
    working_df["patientGender"] = df["Patient Gender"].str.upper().map({"M": 0, "F": 1})

    # Convert view position to binary (case-insensitive)
    working_df["viewPosition"] = df["View Position"].str.upper().map({"PA": 0, "AP": 1})
    label_names = [
        "label_atelectasis",
        "label_cardiomegaly",
        "label_consolidation",
        "label_edema",
        "label_effusion",
        "label_emphysema",
        "label_fibrosis",
        "label_hernia",
        "label_infiltration",
        "label_mass",
        "label_no_finding",
        "label_nodule",
        "label_pleural_thickening",
        "label_pneumonia",
        "label_pneumothorax",
    ]
    # Generate one-hot encoded labels
    for idx, row in df.iterrows():
        labels = generate_image_labels(row["Finding Labels"])
        if idx == 0:  # First iteration, create column names

            for name in label_names:
                working_df[name] = 0

        # Update the label columns for this row
        for col, value in zip(label_names, labels):
            working_df.at[idx, col] = value.item()

    return working_df


def randomize_df(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """
    Randomize the order of rows in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to be randomized.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Randomized DataFrame.
    """
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): Input DataFrame to be split.
        test_size (float): Proportion of the DataFrame to include in the
            test split (0 < test_size < 1).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.

    Raises:
        ValueError: If test_size is not between 0 and 1 exclusive.
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1 exclusive")
    train_df = df.sample(frac=1 - test_size, random_state=seed)
    test_df = df.drop(train_df.index)
    return train_df, test_df
