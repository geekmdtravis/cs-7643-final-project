from pathlib import Path

import pandas as pd

from src.utils import create_working_tabular_df, set_seed, train_test_split


def split_save_data(
    clinical_data: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Splits the clinical data into training and testing sets and saves them to
        the specified directory as 'train.csv' and 'test.csv'.

    Args:
        clinical_data (Path): Path to the clinical data CSV file.
        output_dir (Path): Directory where the split data will be saved.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(random_state)

    # Load and preprocess tabular data
    clinical_df = pd.read_csv(clinical_data)
    _clinical_df = create_working_tabular_df(clinical_df)

    # Split the data into training and testing sets
    split = train_test_split(
        _clinical_df, test_size=test_size, random_state=random_state
    )
    train_df: pd.DataFrame = impute_and_normalize(split[0])
    test_df: pd.DataFrame = impute_and_normalize(split[1])

    # Save the split data to the specified directory
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def impute_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute and normalize values to [0,1] range.
    """
    new_df = df.copy()

    # Imputation with medians
    age_median = new_df["patientAge"].median()
    follow_up_median = new_df["followUpNumber"].median()

    # Handle outliers
    new_df.loc[new_df["patientAge"] > 120, "patientAge"] = age_median
    new_df.loc[new_df["followUpNumber"] > 10, "followUpNumber"] = follow_up_median

    # Min-max scaling with epsilon to avoid division by zero
    eps = 1e-8
    for col in ["patientAge", "followUpNumber"]:
        min_val = new_df[col].min()
        max_val = new_df[col].max()
        new_df[col] = (new_df[col] - min_val) / (max_val - min_val + eps)

    return new_df
