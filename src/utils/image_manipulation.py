"""
Add padding to an image (torch.Tensor) to prepare it for a model.
"""

from typing import Literal
import torch


def pad_image(image: torch.Tensor, padding: int = 16) -> torch.Tensor:
    """
    Add padding to an image (torch.Tensor) to prepare it for a model.

    Args:
        image (torch.Tensor): The input image tensor.
        padding (int): The amount of padding to add to each side of the image.
            Defaults to 16 as that is the recommended patch size for the
            vision transformer model.

    Returns:
        torch.Tensor: The padded image tensor.
    """
    # Check if the input is a 3D tensor (C, H, W)
    if len(image.shape) != 3:
        raise ValueError("pad_image: Input image must be a 3D tensor (C, H, W).")

    # Add padding
    padded_image = torch.nn.functional.pad(
        image, (padding, padding, padding, padding), mode="constant", value=0
    )

    return padded_image


def embed_clinical_data_into_image(
    image: torch.Tensor,
    age: float,
    gender: Literal["male", "female"],
    xr_pos: Literal["AP", "PA"],
    xr_count: int,
    matrix_size: int = 16,
) -> torch.Tensor:
    """
    Embed clinical data into the image tensor. The clinical data is embedded as
    a matrix in the upper-left corner of the image, occupying a total region of
    matrix_size x matrix_size pixels. This region is divided into four quadrants,
    each of size (matrix_size // 2) x (matrix_size // 2). It will be added to all channels
    of the image. The quadrants represent the clinical data as follows:

    - Top-left quadrant (0,0): Scaled age (0-120) to 0-1.
    - Top-right quadrant (0,1): 0 for male, 1 for female.
    - Bottom-left quadrant (1,0): 0 for AP, 1 for PA.
    - Bottom-right quadrant (1,1): Scaled X-ray count (0-10) to 0-1.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        age (float): The age of the patient (0-120).
        gender (Literal): The gender of the patient ('male' or 'female').
        xr_pos (Literal): The position of the X-ray ('AP' or 'PA').
        xr_count (int): The number of X-rays taken (0-10).
        matrix_size (int): The total size (height and width) of the embedded clinical data
            matrix. Must be an even integer. Defaults to 16, matching the ViT patch size.

    Returns:
        torch.Tensor: The image tensor with embedded clinical data in the top-left corner.

    Raises:
        ValueError: If the input image is not a 3D tensor or if the dimensions are too small.
        ValueError: If age is not between 0 and 120.
        ValueError: If xr_count is not greater than 0.
        ValueError: If matrix_size is not a positive, even integer.
    """
    # --- Input Validation ---
    if len(image.shape) != 3:
        raise ValueError(
            "embed_clinical_data_into_image: Input image must be a 3D tensor (C, H, W)."
        )

    if matrix_size <= 0:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must be a positive integer."
        )
    if matrix_size % 2 != 0:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must be an even integer."
        )

    _, height, width = image.shape
    required_dim = matrix_size  # Check against the total embedding size
    if height < required_dim or width < required_dim:
        raise ValueError(
            f"embed_clinical_data_into_image: Image dimensions ({height}x{width}) "
            f"must be at least {required_dim}x{required_dim} to embed data "
            f"with matrix_size={matrix_size}."
        )

    if not (0 <= age <= 120):
        raise ValueError(
            f"embed_clinical_data_into_image: Age ({age}) must be between 0 and 120."
        )

    if xr_count <= 0:
        raise ValueError(
            f"embed_clinical_data_into_image: X-ray count ({xr_count}) must be between 1 and 10."
        )

    if gender not in ["male", "female"]:
        raise ValueError(
            f"embed_clinical_data_into_image: Gender {gender}, must be either 'male' or 'female'."
        )

    if xr_pos not in ["AP", "PA"]:
        raise ValueError(
            f"embed_clinical_data_into_image: X-ray position {xr_pos}, must be either 'AP' or 'PA'."
        )

    # --- Data Scaling ---
    # Clamp values just in case, although validation should catch extremes
    age_scaled = torch.clamp(torch.tensor(age / 120.0), 0.0, 1.0)
    gender_val = torch.tensor(0.0) if gender == "male" else torch.tensor(1.0)
    xr_pos_val = torch.tensor(0.0) if xr_pos == "AP" else torch.tensor(1.0)
    xr_count_scaled = torch.clamp(torch.tensor(xr_count / 10.0), 0.0, 1.0)

    # --- Embedding ---
    # Create a copy to avoid modifying the original tensor
    image_copy = image.clone()
    quad_size = matrix_size // 2

    # Embed Age (Top-left quadrant)
    image_copy[:, 0:quad_size, 0:quad_size] = age_scaled

    # Embed Gender (Top-right quadrant)
    image_copy[:, 0:quad_size, quad_size:matrix_size] = gender_val

    # Embed X-ray Position (Bottom-left quadrant)
    image_copy[:, quad_size:matrix_size, 0:quad_size] = xr_pos_val

    # Embed X-ray Count (Bottom-right quadrant)
    image_copy[:, quad_size:matrix_size, quad_size:matrix_size] = xr_count_scaled

    return image_copy
