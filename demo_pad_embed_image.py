"""
Demo padding and embedding of clinical data into images.
"""

import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from src.data import download_dataset
from src.utils import pad_image, embed_clinical_data_into_image, Config

cfg = Config()

paths = download_dataset()

_test_image_1 = Image.open(paths.images_dir / "00000001_000.png")
_test_image_2 = Image.open(paths.images_dir / "00000001_001.png")
_test_image_3 = Image.open(paths.images_dir / "00000001_002.png")
test_image_1 = TF.to_tensor(_test_image_1)
test_image_2 = TF.to_tensor(_test_image_2)
test_image_3 = TF.to_tensor(_test_image_3)

batched_images = torch.stack([test_image_1, test_image_2, test_image_3], dim=0)

padded_image = pad_image(images=batched_images, padding=0)
embedded_image = embed_clinical_data_into_image(
    image=padded_image,
    age=[25, 30, 99],
    gender=["male", "female", "female"],
    xr_pos=["AP", "PA", "PA"],
    xr_count=[1, 3, 1],
    matrix_size=16,
)


# Create output directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# Convert tensors to PIL Images and save
# First, we need to convert from [0,1] range to [0,255] for PIL
# and move tensor to CPU if it's on GPU
img_tensor = embedded_image.cpu().detach()

# Save each image in the batch
for idx in range(img_tensor.shape[0]):
    # Convert to PIL Image (assuming the tensor is in [B, C, H, W] format)
    pil_image = TF.to_pil_image(img_tensor[idx])
    # Save as PNG with unique name for each image
    path = os.path.join("artifacts", f"embedded_image_{idx+1}.png")
    pil_image.save(path)
    print(f"Embedded image {idx+1} saved to {path}")
