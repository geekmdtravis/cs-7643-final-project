from pathlib import Path

import torch
import torchvision.transforms as T

from src.data import create_dataloader
from src.utils import Config, get_system_info

cfg = Config()


def save_tensor_as_image(tensor: torch.Tensor, save_path: Path):
    """Convert a tensor to a PIL Image and save it."""
    # ToPILImage expects CxHxW in [0, 1] range for best results without denormalization
    # Tensors directly from 'none' normalization mode should be fine.
    # Tensors from other modes might look strange without denormalization,
    # but this script aims to show the *effect* of the normalization itself.
    to_pil = T.ToPILImage()
    img = to_pil(tensor.cpu())  # Ensure tensor is on CPU
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    print(f"Saved image: {save_path.name}")


def process_and_save_batch(
    loader: torch.utils.data.DataLoader, norm_mode_suffix: str, demo_dir: Path
):
    """Processes the first batch of a loader and saves images."""
    print(f"\nProcessing first batch with normalization: '{norm_mode_suffix}'...")
    try:
        # Get the first batch
        batch_data = next(iter(loader))
        cxr: torch.Tensor = batch_data[0]
        tabular: torch.Tensor = batch_data[1]
        labels: torch.Tensor = batch_data[2]

        print(f"- CXR shape: {cxr.shape}")
        print(f"- Clinical data shape: {tabular.shape}")
        print(f"- Labels shape: {labels.shape}")

        # Save each image in the batch
        for j in range(cxr.shape[0]):
            save_path = demo_dir / f"batch_1_img_{j+1}_norm_{norm_mode_suffix}.png"
            save_tensor_as_image(cxr[j], save_path)
            if j >= 9:  # Limit saving to first 10 images per batch for demo speed
                print("  (Reached demo limit of 10 images for this batch)")
                break

    except StopIteration:
        print("Loader is empty.")
    except Exception as e:
        print(f"An error occurred processing batch: {e}")


def main():
    print("Initializing demo...")
    sys_info = get_system_info()
    print(sys_info)

    # Common dataloader settings for the demo
    # Use batch_size=10 to match the image saving limit easily
    # Set shuffle=False to get consistent images for comparison
    loader_args = {
        "clinical_data": cfg.clinical_data,
        "cxr_images_dir": cfg.cxr_train_dir,
        "batch_size": 10,
        "num_workers": 0,  # Use 0 workers for simplicity and consistency in demo
    }

    # --- Process with 'none' normalization ---
    loader_none = create_dataloader(**loader_args, normalization_mode="none")
    process_and_save_batch(loader_none, "none", cfg.demo_dir)

    # --- Process with 'dataset_specific' normalization ---
    loader_dataset = create_dataloader(
        **loader_args, normalization_mode="dataset_specific"
    )
    process_and_save_batch(loader_dataset, "dataset", cfg.demo_dir)

    # --- Process with 'imagenet' normalization ---
    loader_imagenet = create_dataloader(**loader_args, normalization_mode="imagenet")
    process_and_save_batch(loader_imagenet, "imagenet", cfg.demo_dir)

    print("\nDemo finished. Check the 'artifacts/demo' directory for output images.")


if __name__ == "__main__":
    main()
