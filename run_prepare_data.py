"""Example usage of the ChestXrayDataset and dataloaders."""

from src.data import prepare_data


def main():
    """Demonstrate dataset usage."""
    print(
        "Preparing data.\n"
        " - Splitting data into train, validation, and test sets.\n"
        " - Saving the splits to disk.\n"
        " - Creating clinical matrix-embedded images for each split.\n"
        " - Saving the images to disk.\n\n"
        "Upon completion, check the 'artifacts' directory for the files "
        "you will load via the create_dataloader function.\n\n"
    )
    prepare_data(test_size=0.15, val_size=0.15)


if __name__ == "__main__":
    main()
