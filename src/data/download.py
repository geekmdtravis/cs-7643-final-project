from pathlib import Path
from dataclasses import dataclass
import sys
import kagglehub


@dataclass
class FilePaths:
    """
    Class to hold file paths for the NIH Chest X-ray dataset.
    """

    csv_dir: Path | None = None
    images_dir: Path | None = None
    train_val_list: Path | None = None
    test_list: Path | None = None

    def __post_init__(self):
        if self.csv_dir is None:
            self.csv_dir = Path()
        if self.images_dir is None:
            self.images_dir = Path()
        if self.train_val_list is None:
            self.train_val_list = Path()
        if self.test_list is None:
            self.test_list = Path()

    def __repr__(self):
        return f"FilePaths(csv_dir={self.csv_dir}, images_dir={self.images_dir}, train_val_list={self.train_val_list}, test_list={self.test_list})"

    def __str__(self):
        return f"FilePaths(\n\tcsv_dir={self.csv_dir}, \n\timages_dir={self.images_dir}, \n\ttrain_val_list={self.train_val_list}, \n\ttest_list={self.test_list})"


def download_dataset() -> FilePaths:
    """
    Download the NIH Chest X-ray dataset using kagglehub.

    Args:
        target_dir: Directory where the dataset should be stored

    Returns:
        FilePaths: A dataclass containing paths to the dataset files.
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
        # Create target directory if it doesn't exist
        target_path = Path()
        target_path.mkdir(parents=True, exist_ok=True)

        print("Downloading NIH Chest X-ray dataset...")
        # Download dataset
        # Download dataset using the default cache location
        # Removing the 'path' argument resolved the download issue.
        cache_path = kagglehub.dataset_download(
            "khanfashee/nih-chest-x-ray-14-224x224-resized"
        )

    except (IOError, OSError) as e:
        print(f"Error with file operations: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid value error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Runtime error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # Define file paths
    cache_path = Path(cache_path)
    csv_dir = cache_path / "Data_Entry_2017.csv"
    images_dir = cache_path / "images-224/images-224/"
    train_val_list = cache_path / "train_val_list_NIH.txt"
    test_list = cache_path / "test_list_NIH.txt"
    file_paths = FilePaths(
        csv_dir=csv_dir,
        images_dir=images_dir,
        train_val_list=train_val_list,
        test_list=test_list,
    )
    return file_paths


if __name__ == "__main__":
    paths = download_dataset()
    print(paths)
