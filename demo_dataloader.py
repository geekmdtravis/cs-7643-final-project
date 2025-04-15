from src.data import create_dataloader
from src.utils.path_utils import get_project_root


def main():
    print("Loading chest X-ray dataset...")
    root = get_project_root()
    train_cxrs_dir = root / "artifacts" / "embedded_train"
    train_tabular = root / "artifacts" / "train.csv"

    loader = create_dataloader(
        clinical_data=train_tabular,
        cxr_images_dir=train_cxrs_dir,
        batch_size=32,
        num_workers=4,
    )

    print("\nBatch contents preview (showing first 3 batches):")
    for i, (cxr, tabular, labels) in enumerate(loader):
        print(f"\nBatch {i + 1}:")
        print(f"- CXR shape: {cxr.shape}")
        print(f"- Clinical data shape: {tabular.shape}")
        print(f"- Labels shape: {labels.shape}")

        if i >= 2:  # Only show first 3 batches
            break


if __name__ == "__main__":
    main()
