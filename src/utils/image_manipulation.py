"""
Add padding to an image (torch.Tensor) to prepare it for a model and embed
clinical data into images.
"""

from typing import List, Literal, Union

import torch


def pad_image(images: torch.Tensor, padding: int = 16) -> torch.Tensor:
    """
    Add padding to an image or batch of images (torch.Tensor) to prepare for a model.

    Args:
        images (torch.Tensor): The input image tensor.
            Can be either:
            - Single image: (C, H, W)
            - Batch of images: (B, C, H, W)
        padding (int): The amount of padding to add to each side of the image.
            Defaults to 16 as that is the recommended patch size for the
            vision transformer model.

    Returns:
        torch.Tensor: The padded image tensor with same number of dimensions as input.
    """
    # Check if input is valid (3D or 4D tensor)
    if len(images.shape) not in [3, 4]:
        raise ValueError(
            "pad_image: Input must be a 3D tensor (C, H, W) or 4D tensor (B, C, H, W)."
        )

    # Add padding
    padded_image = torch.nn.functional.pad(
        images, (padding, padding, padding, padding), mode="constant", value=0
    )

    return padded_image


def embed_clinical_data_into_image(
    image: torch.Tensor,
    age: Union[float, List[float]],
    gender: Union[Literal["male", "female"], List[Literal["male", "female"]]],
    xr_pos: Union[Literal["AP", "PA"], List[Literal["AP", "PA"]]],
    xr_count: Union[int, List[int]],
    matrix_size: int = 16,
) -> torch.Tensor:
    """
    Embed clinical data into the image tensor(s). The clinical data is embedded as
    a matrix in the upper-left corner of each image, occupying a total region of
    matrix_size x matrix_size pixels. This region is divided into four quadrants,
    each of size (matrix_size // 2) x (matrix_size // 2). It will be added to
    all channels of the image. The quadrants represent the clinical data as follows:

    - Top-left quadrant (0,0): Scaled age (0-120) to 0-1.
    - Top-right quadrant (0,1): 0 for male, 1 for female.
    - Bottom-left quadrant (1,0): 0 for AP, 1 for PA.
    - Bottom-right quadrant (1,1): Scaled X-ray count (0-10) to 0-1.

    Args:
        image (torch.Tensor): The input image tensor, either (C, H, W) for a
            single image or (B, C, H, W) for a batch of images.
        age (Union[float, List[float]]): Age(s) of the patient(s) (0-120).
        gender (Union[Literal, List[Literal]]): Gender(s) of the
            patient(s) ('male' or 'female').
        xr_pos (Union[Literal, List[Literal]]): Position(s) of the
            X-ray(s) ('AP' or 'PA').
        xr_count (Union[int, List[int]]): Number(s) of X-rays taken (0-10).
        matrix_size (int): The total size (height and width) of the embedded
            clinical data matrix. Must be an even integer. Defaults to 16,
            matching the ViT patch size.

    Returns:
        torch.Tensor: The image tensor with embedded clinical data
            in the top-left corner. Will have the same number of dimensions as input.

    Raises:
        ValueError: If the input image dimensions are invalid or too small.
        ValueError: If age values are not between 0 and 120.
        ValueError: If xr_count values are not greater than 0.
        ValueError: If matrix_size is not a positive, even integer.
        ValueError: If batch sizes don't match between image and clinical data.
    """
    # --- Input Validation ---
    if len(image.shape) not in [3, 4]:
        raise ValueError(
            "embed_clinical_data_into_image: Input must be a 3D tensor (C, H, W) "
            "or 4D tensor (B, C, H, W)."
        )

    if matrix_size <= 0:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must "
            f"be a positive integer."
        )
    if matrix_size % 2 != 0:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must "
            f"be an even integer."
        )

    # Determine if we're working with a batch
    is_batch = len(image.shape) == 4
    batch_size = image.shape[0] if is_batch else 1

    # Validate image dimensions
    if is_batch:
        _, channels, height, width = image.shape
    else:
        channels, height, width = image.shape

    required_dim = matrix_size
    if height < required_dim or width < required_dim:
        raise ValueError(
            f"embed_clinical_data_into_image: Image dimensions ({height}x{width}) "
            f"must be at least {required_dim}x{required_dim} to embed data "
            f"with matrix_size={matrix_size}."
        )

    # For batch operations, require lists that match batch size
    if is_batch:
        if (
            not isinstance(age, list)
            or not isinstance(gender, list)
            or not isinstance(xr_pos, list)
            or not isinstance(xr_count, list)
        ):
            raise ValueError(
                "Batch operations require lists for all clinical data parameters"
            )
        ages = age
        genders = gender
        xr_positions = xr_pos
        xr_counts = xr_count
    else:
        # Single image case
        ages = [age]
        genders = [gender]
        xr_positions = [xr_pos]
        xr_counts = [xr_count]

    # Validate clinical data batch sizes
    if len(ages) != batch_size:
        raise ValueError(
            f"Number of ages ({len(ages)}) must match batch size ({batch_size})"
        )
    if len(genders) != batch_size:
        raise ValueError(
            f"Number of genders ({len(genders)}) must match batch size ({batch_size})"
        )
    if len(xr_positions) != batch_size:
        raise ValueError(
            f"Number of X-ray positions ({len(xr_positions)}) must match "
            f"batch size ({batch_size})"
        )
    if len(xr_counts) != batch_size:
        raise ValueError(
            f"Number of X-ray counts ({len(xr_counts)}) must match "
            f"batch size ({batch_size})"
        )

    # Validate clinical data values
    for i, (a, g, pos, count) in enumerate(zip(ages, genders, xr_positions, xr_counts)):
        if not (0 <= a <= 120):
            raise ValueError(
                f"embed_clinical_data_into_image: Age ({a}) at index {i} must "
                f"be between 0 and 120."
            )
        if count < 0:
            raise ValueError(
                f"embed_clinical_data_into_image: X-ray follow-up ({count}) "
                f"at index {i} must be greater than or equal to 0."
            )
        if g not in ["male", "female"]:
            raise ValueError(
                f"embed_clinical_data_into_image: Gender {g} at index {i}, "
                f"must be either 'male' or 'female'."
            )
        if pos not in ["AP", "PA"]:
            raise ValueError(
                f"embed_clinical_data_into_image: X-ray position {pos} at "
                f"index {i}, must be either 'AP' or 'PA'."
            )

    # --- Data Scaling ---
    age_scaled = torch.tensor([a / 120.0 for a in ages]).clamp(0.0, 1.0)
    gender_val = torch.tensor([0.0 if g == "male" else 1.0 for g in genders])
    xr_pos_val = torch.tensor([0.0 if pos == "AP" else 1.0 for pos in xr_positions])
    xr_count_scaled = torch.tensor([c / 10.0 for c in xr_counts]).clamp(0.0, 1.0)

    # --- Embedding ---
    # Create a copy to avoid modifying the original tensor
    image_copy = image.clone()
    quad_size = matrix_size // 2

    # Convert to batch format if single image
    if not is_batch:
        image_copy = image_copy.unsqueeze(0)
        age_scaled = age_scaled.unsqueeze(0)
        gender_val = gender_val.unsqueeze(0)
        xr_pos_val = xr_pos_val.unsqueeze(0)
        xr_count_scaled = xr_count_scaled.unsqueeze(0)

    # Reshape clinical data for broadcasting to match batch, height, width dimensions
    age_scaled = age_scaled.view(-1, 1, 1).expand(-1, quad_size, quad_size)
    gender_val = gender_val.view(-1, 1, 1).expand(-1, quad_size, quad_size)
    xr_pos_val = xr_pos_val.view(-1, 1, 1).expand(-1, quad_size, quad_size)
    xr_count_scaled = xr_count_scaled.view(-1, 1, 1).expand(-1, quad_size, quad_size)

    # Embed values for the entire batch
    for i in range(channels):
        # Embed each value in the corresponding quadrant for all images in the batch
        image_copy[:, i, 0:quad_size, 0:quad_size] = age_scaled
        image_copy[:, i, 0:quad_size, quad_size:matrix_size] = gender_val
        image_copy[:, i, quad_size:matrix_size, 0:quad_size] = xr_pos_val
        image_copy[:, i, quad_size:matrix_size, quad_size:matrix_size] = xr_count_scaled

    # Convert back to single image format if input was single image
    if not is_batch:
        image_copy = image_copy.squeeze(0)

    return image_copy
