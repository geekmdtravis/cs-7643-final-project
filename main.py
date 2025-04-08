"""This script is used to run the main functionality of the application."""

import pandas as pd
from src.utils import (
    get_system_info,
    randomize_df,
    create_working_tabular_df,
    set_seed,
    train_test_split,
)
from src.data import download_dataset

SEED = 42

artifacts_dir = "artifacts"


def print_intro():
    """Prints the introduction of the application."""
    print("===============================================")
    print("CS7643 Final Project - Multimodal Chest X-ray Classification")
    print("-----------------------------------------------")
    print(f"System Information:\n{get_system_info()}")
    print("-----------------------------------------------")


if __name__ == "__main__":
    print_intro()
    paths = download_dataset()
    set_seed(SEED)
    _df = pd.read_csv(paths.clinical_data)
    _ppdf = create_working_tabular_df(_df)
    df = randomize_df(_ppdf)

    df_train, df_test = train_test_split(df, test_size=0.2, seed=SEED)
    print("Saving train and test data to CSV files...")
    df_train.to_csv("./artifacts/train.csv", index=False)
    df_test.to_csv("./artifacts/test.csv", index=False)
    print("Train and test data saved successfully.")
