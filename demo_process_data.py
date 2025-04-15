"""Example usage of the ChestXrayDataset and dataloaders."""

from src.data import copy_csv_to_artifacts, copy_images_to_artifacts, download_dataset


def main():
    """Demonstrate dataset usage."""
    file_paths = download_dataset()

    copy_csv_to_artifacts(file_paths.clinical_data)
    copy_images_to_artifacts(file_paths.images_dir)


if __name__ == "__main__":
    main()
