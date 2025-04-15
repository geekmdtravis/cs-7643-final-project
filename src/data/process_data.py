"""
Images downloaded via Kaggle Hub will need to be
prepared for local use. There are two forms each image
must take: (1) The original, unmodified image, and
(2) a version with clinical tabular data embedded as
a matrix into the upper-left corner of the image.
This script will copy the origimal images to
{workspace}/artifacts/original_images and the embedded
images to {workspace}/artifacts/embedded_images.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import download_dataset
from src.utils import create_working_tabular_df, set_seed, train_test_split
from src.utils.path_utils import get_project_root


def __split_save_data(
    clinical_data: Path,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Splits the clinical data into training and testing sets and saves them to
        the specified directory as 'train.csv' and 'test.csv'.

    Args:
        clinical_data (Path): Path to the clinical data CSV file.
        output_dir (Path): Directory where the split data will be saved.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Load and preprocess tabular data
    clinical_df = pd.read_csv(clinical_data)
    _clinical_df = create_working_tabular_df(clinical_df)

    # Split the data into training and testing sets
    split = train_test_split(_clinical_df, test_size=test_size, seed=seed)
    train_df: pd.DataFrame = __impute_and_normalize(split[0])
    test_df: pd.DataFrame = __impute_and_normalize(split[1])

    # Save the split data to the specified directory
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def __impute_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
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


def __copy_images_to_artifacts(
    clinical_data: Path,
    kaggle_cache_cxr_dir: Path,
    dest_dir: Path | None = None,
) -> None:
    """
    For the CXR files identified in the clinical data,
    copy the images from the Kaggle cache to the
    {workspace}/artifacts/cxr_images directory. The clinical data
    being pointed to is expected to be the one that was
    already split into train and test sets.

    This is useful for preparing the images for local
    processing after it has been split into train and test
    sets.

    Args:
        clinical_data (Path): Path to the clinical data CSV file.
        kaggle_cache_cxr_dir (Path): Path to the Kaggle cache directory
            containing CXR images.
        dest_dir (Path | None): Directory where the images will be
            copied. If None, defaults to
            {workspace}/artifacts/cxr_images.

    Returns:
        None
    """
    if dest_dir is None:
        project_root = get_project_root()
        artifacts_dir = project_root / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir.mkdir(parents=True)
        dest_dir = artifacts_dir / "cxr_images"

    if not kaggle_cache_cxr_dir.exists():
        raise FileNotFoundError(f"{kaggle_cache_cxr_dir} does not exist.")
    if not kaggle_cache_cxr_dir.is_dir():
        raise NotADirectoryError(f"{kaggle_cache_cxr_dir} is not a directory.")
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"{dest_dir} is not a directory.")

    clinical_df = pd.read_csv(clinical_data)
    image_list = list(kaggle_cache_cxr_dir.glob("*.png"))
    for image in tqdm(image_list, desc="Copying images"):
        if image.name not in clinical_df["imageIndex"].values:
            continue
        if image.is_file():
            dest_image = dest_dir / image.name
            dest_image.write_bytes(image.read_bytes())
        else:
            raise FileNotFoundError(f"{image} is not a file.")
    print(f"Copied images from {kaggle_cache_cxr_dir} to {dest_dir}.")


def copy_csv_to_artifacts(
    origin_file: Path,
    dest_file: Path | None = None,
) -> None:
    """
    The origin file is the csv within the Kaggle
    cached dataset. The destination file will default to
    {workspace}/artifacts/tabular_data.csv.
    """
    if dest_file is None:
        project_root = get_project_root()
        artifacts_dir = project_root / "artifacts"
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"{artifacts_dir} does not exist.")
        dest_dir = artifacts_dir
        dest_file = dest_dir / "clinical_data.csv"

    if not origin_file.exists():
        raise FileNotFoundError(f"{origin_file} does not exist.")
    if not origin_file.is_file():
        raise FileNotFoundError(f"{origin_file} is not a file or was not found.")
    if not dest_dir.exists():
        dest_dir.mkdir()
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"{dest_dir} is not a directory.")

    dest_file.write_bytes(origin_file.read_bytes())
    print(f"Copied csv from {origin_file} to {dest_file}.")


def process_data(test_size: float = 0.2, seed: int = 42) -> None:
    """
    Process the data for the Chest X-ray dataset. The
    output directory of the clinical data is
    `{workspace}/artifacts/{train|test}.csv` and the
    images are saved to `{workspace}/artifacts/cxr_images` and
    `{workspace}/artifacts/embedded_images`.
    """

    clinical_data, cxr_images_dir = download_dataset()

    root = get_project_root()
    __split_save_data(
        clinical_data=clinical_data,
        output_dir=root / "artifacts",
        test_size=test_size,
        seed=seed,
    )

    test_data = root / "artifacts" / "test.csv"
    train_data = root / "artifacts" / "train.csv"

    __copy_images_to_artifacts(
        clinical_data=train_data,
        kaggle_cache_cxr_dir=cxr_images_dir,
        dest_dir=root / "artifacts" / "cxr_images_train",
    )
    __copy_images_to_artifacts(
        clinical_data=test_data,
        kaggle_cache_cxr_dir=cxr_images_dir,
        dest_dir=root / "artifacts" / "cxr_images_test",
    )
