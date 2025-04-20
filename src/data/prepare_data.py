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

import logging
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.data import download_dataset
from src.utils import (
    Config,
    create_working_tabular_df,
    embed_clinical_data_into_image,
    set_seed,
    train_test_split,
)

cfg = Config()


def shuffle_split_save(
    clinical_data: Path,
    output_dir: Path,
    val_size: float,
    test_size: float,
    seed: int = 42,
):
    """
    Splits the clinical data into training, validation, and testing
    sets and saves them to the specified directory as 'train.csv',
    'val.csv', and 'test.csv'.

    Args:
        clinical_data (Path): Path to the clinical data CSV file.
        output_dir (Path): Directory where the split data will be saved.
        val_size (float): Proportion of non-test data to include in validation split.
        test_size (float): Proportion of the entire dataset to include in test split.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Load and preprocess tabular data
    clinical_df = pd.read_csv(clinical_data)
    _clinical_df = create_working_tabular_df(clinical_df)

    # First split off test set
    train_val_df, test_df = train_test_split(
        _clinical_df, test_size=test_size, seed=seed
    )

    # Calculate val_size relative to remaining data to maintain desired ratio
    effective_val_size = val_size * (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=effective_val_size, seed=seed
    )

    # Apply normalization to all splits
    train_df = impute_and_normalize(train_df)
    val_df = impute_and_normalize(val_df)
    test_df = impute_and_normalize(test_df)

    if not output_dir.exists():
        error = f"shuffle_split_save: output_dir={output_dir} does not exist"
        logging.error(error)
        raise FileNotFoundError(error)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def impute_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing or outlier data, and normalize values to [0,1] range.

    Returns:
        pd.DataFrame: DataFrame with imputed and normalized values as
            a new DataFrame.
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


def copy_cached_imgs_to_artifacts(
    clinical_data: Path,
    kaggle_cache_cxr_dir: Path,
    dest_dir: Path,
) -> None:
    """
    For the CXR files identified in the clinical data,
    copy the images from the Kaggle cache to the
    {workspace}/artifacts/{dest_dir}. The clinical data
    being pointed to is expected to be the one that was
    already split into train and test sets.

    This is useful for preparing the images for local
    processing after it has been split into train and test
    sets.

    Args:
        clinical_data (Path): Path to the clinical data CSV file.
        kaggle_cache_cxr_dir (Path): Path to the Kaggle cache directory
            containing CXR images.
        dest_dir (Path ): Directory where the images will be
            copied.

    Returns:
        None
    """
    if not dest_dir.exists():
        error = f"copy_cached_imgs_to_artifacts: dest_dir={dest_dir} does not exist"
        logging.error(error)
        raise FileNotFoundError(error)

    if not kaggle_cache_cxr_dir.exists():
        error = (
            "copy_cached_imgs_to_artifacts: given kaggle cache "
            f"directory {kaggle_cache_cxr_dir} does not exist."
        )
        logging.error(error)
        raise FileNotFoundError(error)

    if not kaggle_cache_cxr_dir.is_dir():
        error = (
            f"copy_cached_imgs_to_artifacts: {kaggle_cache_cxr_dir} is not a directory."
        )
        logging.error(error)
        raise NotADirectoryError(error)

    clinical_df = pd.read_csv(clinical_data)
    # Get set of valid image indices for faster lookup
    valid_images = set(clinical_df["imageIndex"].values)
    # Filter image list before processing
    image_list = [
        img for img in kaggle_cache_cxr_dir.glob("*.png") if img.name in valid_images
    ]

    for image in tqdm(image_list, desc="Copying images"):
        if image.is_file():
            dest_image = dest_dir / image.name
            dest_image.write_bytes(image.read_bytes())
        else:
            raise FileNotFoundError(f"{image} is not a file.")
    logging.info(
        "copy_cached_imgs_to_artifacts: Copied images "
        f"from {kaggle_cache_cxr_dir} to {dest_dir}."
    )


def create_save_embedded_images(
    images_dir: Path, clinical_data: Path, dest_dir: Path
) -> None:
    """
    Copy the embedded images to the artifacts directory.
    """
    if not dest_dir.exists():
        error = f"create_save_embedded_images: dest_dir={dest_dir} does not exist"
        logging.error(error)
        raise FileNotFoundError(error)

    clinical_df = pd.read_csv(clinical_data)

    images_list = list(images_dir.glob("*.png"))

    for image in tqdm(images_list, desc="Copying embedded images"):
        if image.is_file():
            image_name = image.name
            pil_image = Image.open(image)
            image_tensor = TF.to_tensor(pil_image)
            # Get matching clinical data row
            matching_row = clinical_df[clinical_df["imageIndex"] == image_name]
            if matching_row.empty:
                error = (
                    "create_save_embedded_images: No "
                    f"clinical data found for {image_name}"
                )
                logging.error(error)
                raise ValueError(error)

            tabular_batch = torch.tensor(
                matching_row.iloc[0].values[1:5].astype(float)
            ).float()

            embedded_img = embed_clinical_data_into_image(
                image_batch=torch.stack([image_tensor]),
                tabular_batch=torch.stack([tabular_batch]),
                matrix_size=16,
            )
            save_path = dest_dir / image_name
            embedded_image = embedded_img.cpu().detach()[0]
            embedded_pil_image = TF.to_pil_image(embedded_image)
            embedded_pil_image.save(save_path)

        else:
            error = f"create_save_embedded_images: {image} is not a file."
            logging.error(error)
            raise FileNotFoundError(error)


def prepare_data(test_size: float, val_size: float, seed: int = 42) -> None:
    """
    Process the data for the Chest X-ray dataset. The
    output directory of the clinical data is
    `{workspace}/artifacts/{train|val|test}.csv` and the
    images are saved to `{workspace}/artifacts/cxr_{train|val|test}` and
    `{workspace}/artifacts/embedded_{train|val|test}`.

    Args:
        test_size (float): Proportion of entire dataset to use for test split.
        val_size (float): Proportion of non-test data to use for validation split.
        seed (int): Random seed for reproducibility.
    """

    cached_clin_data, cached_cxr_imgs_dir = download_dataset()

    logging.info("process_data: Saving processed clinical data to artifacts directory.")
    shuffle_split_save(
        clinical_data=cached_clin_data,
        output_dir=cfg.artifacts,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    logging.info(
        "process_data: Copying unmodified train images to artifacts directory."
    )
    copy_cached_imgs_to_artifacts(
        clinical_data=cfg.tabular_clinical_train,
        kaggle_cache_cxr_dir=cached_cxr_imgs_dir,
        dest_dir=cfg.cxr_train_dir,
    )
    logging.info(
        "process_data: Copying unmodified validation images to artifacts directory."
    )
    copy_cached_imgs_to_artifacts(
        clinical_data=cfg.tabular_clinical_val,
        kaggle_cache_cxr_dir=cached_cxr_imgs_dir,
        dest_dir=cfg.cxr_val_dir,
    )
    logging.info("process_data: Copying unmodified test images to artifacts directory.")
    copy_cached_imgs_to_artifacts(
        clinical_data=cfg.tabular_clinical_test,
        kaggle_cache_cxr_dir=cached_cxr_imgs_dir,
        dest_dir=cfg.cxr_test_dir,
    )

    logging.info("process_data: Copying embedded train images to artifacts directory.")
    create_save_embedded_images(
        images_dir=cfg.cxr_train_dir,
        clinical_data=cfg.tabular_clinical_train,
        dest_dir=cfg.embedded_train_dir,
    )
    logging.info(
        "process_data: Copying embedded validation images to artifacts directory."
    )
    create_save_embedded_images(
        images_dir=cfg.cxr_val_dir,
        clinical_data=cfg.tabular_clinical_val,
        dest_dir=cfg.embedded_val_dir,
    )
    logging.info("process_data: Copying embedded test images to artifacts directory.")
    create_save_embedded_images(
        images_dir=cfg.cxr_test_dir,
        clinical_data=cfg.tabular_clinical_test,
        dest_dir=cfg.embedded_test_dir,
    )
    logging.info("process_data: Data processing complete.")
