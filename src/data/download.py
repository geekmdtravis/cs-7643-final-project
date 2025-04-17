"""
Download the NIH Chest X-ray dataset using kagglehub.
This script downloads the dataset and provides a dataclass to hold the file paths.
"""

import logging
import sys
from pathlib import Path

import kagglehub


def download_dataset() -> tuple[Path, Path]:
    """
    Download the NIH Chest X-ray dataset using kagglehub and
    return the paths to the clinical data CSV file and images directory.
    The dataset is downloaded to a cache directory managed by kagglehub.

    Returns:
        tuple: Paths to the clinical data CSV file and images directory
              as a tuple (clinical_data, images_dir)
    Raises:
        IOError: If there is an issue with file operations.
        OSError: If there is an OS-related error.
        ValueError: If there is an invalid value error.
        RuntimeError: If there is a runtime error downloading the dataset.
        Exception: For any other unexpected errors.

    Citations:
    - https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized
    """
    try:
        logging.info("download_dataset: Downloading NIH Chest X-ray dataset...")
        cache_path = kagglehub.dataset_download(
            "khanfashee/nih-chest-x-ray-14-224x224-resized"
        )

    except (IOError, OSError) as e:
        logging.error(
            f"download_dataset: Error with file operations: {e}", file=sys.stderr
        )
        sys.exit(1)
    except ValueError as e:
        logging.error(f"download_dataset: Invalid value error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        logging.error(
            f"download_dataset: Runtime error downloading dataset: {e}", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            f"download_dataset: An unexpected error occurred: {e}", file=sys.stderr
        )
        sys.exit(1)

    cache_path = Path(cache_path)
    clinical_data = cache_path / "Data_Entry_2017.csv"
    images_dir = cache_path / "images-224/images-224/"
    return clinical_data, images_dir


if __name__ == "__main__":
    paths = download_dataset()
    print(paths)
