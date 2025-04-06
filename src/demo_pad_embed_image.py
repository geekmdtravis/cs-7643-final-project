"""
Demo padding and embedding of clinical data into images.
"""

import os
from PIL import Image
import torchvision.transforms.functional as TF
from data import download_dataset
from utils import pad_image, embed_clinical_data_into_image

paths = download_dataset()

test_image = Image.open(paths.images_dir / "00000001_001.png")
test_image_tensor = TF.to_tensor(test_image)

padded_image = pad_image(image=test_image_tensor, padding=0)
embedded_image = embed_clinical_data_into_image(
    image=padded_image,
    age=37,
    gender="female",
    xr_pos="AP",
    xr_count=5,
    matrix_size=16,
)


# Create output directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# Convert tensor to PIL Image and save
# First, we need to convert from [0,1] range to [0,255] for PIL
# and move tensor to CPU if it's on GPU
img_tensor = embedded_image.cpu().detach()
# Convert to PIL Image (assuming the tensor is in [C, H, W] format)
pil_image = TF.to_pil_image(img_tensor)
# Save as PNG
path = os.path.join("artifacts", "embedded_image.png")
pil_image.save(path)

print(f"Embedded image saved to {path}")
