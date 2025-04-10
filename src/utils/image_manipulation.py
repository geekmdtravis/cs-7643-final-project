"""
Add padding to an image (torch.Tensor) to prepare it for a model and embed
clinical data into images.
"""

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
    image_batch: torch.Tensor,
    tabular_batch: torch.Tensor,
    matrix_size: int = 16,
) -> torch.Tensor:
    """
    Embed clinical data into the image tensor(s). The clinical data is embedded as
    a matrix in the upper-left corner of each image, occupying a total region of
    matrix_size x matrix_size pixels. This region is divided into four quadrants,
    each of size (matrix_size // 2) x (matrix_size // 2). It will be added to
    all channels of the image. The quadrants represent the clinical data as follows:

    - Top-left quadrant (0,0): Follow-up number (normalized to [0,1])
    - Top-right quadrant (0,1): Patient age (normalized to [0,1])
    - Bottom-left quadrant (1,0): Patient gender (0 for male, 1 for female)
    - Bottom-right quadrant (1,1): View position (0 for AP, 1 for PA)

    Args:
        image_batch (torch.Tensor): The input image tensor (B, C, H, W)
        tabular_batch (torch.Tensor): The input tabular data tensor (B, 4) where:
            - [:, 0]: Normalized follow-up number values in [0,1]
            - [:, 1]: Normalized patient age values in [0,1]
            - [:, 2]: Patient gender values (0 for male, 1 for female)
            - [:, 3]: View position values (0 for AP, 1 for PA)
        matrix_size (int): Size of embedding region. Must be even. Default: 16

    Returns:
        torch.Tensor: The image tensor with embedded clinical data
            in the top-left corner. Will have the same number of dimensions as input.

    Raises:
        ValueError: If the input image dimensions are invalid or too small.
        ValueError: If matrix_size is not a positive, even integer.
        ValueError: If batch sizes don't match between image and clinical data.
    """
    img_batch_size = image_batch.shape[0]
    tabular_batch_size = tabular_batch.shape[0]
    if img_batch_size != tabular_batch_size:
        raise ValueError(
            f"embed_clinical_data_into_image: Batch sizes "
            f"must match. Image batch size: "
            f"{img_batch_size}, Tabular batch size: {tabular_batch_size}"
        )
    img_width = image_batch.shape[-1]
    img_height = image_batch.shape[-2]
    if img_width < matrix_size or img_height < matrix_size:
        raise ValueError(
            f"embed_clinical_data_into_image: Image "
            f"dimensions ({img_height}x{img_width}) "
            f"must be at least {matrix_size}x{matrix_size} to embed data "
            f"with matrix_size={matrix_size}."
        )

    if img_width < 32 or img_height < 32:
        raise ValueError(
            f"embed_clinical_data_into_image: Image "
            f"dimensions ({img_height}x{img_width}) "
            f"must be at least 32x32 to embed data with matrix_size={matrix_size}."
        )

    if matrix_size % 2 != 0:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must be "
            f"an even integer."
        )
    if matrix_size <= 0:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must "
            f"be a positive integer."
        )

    if matrix_size > img_width // 2 or matrix_size > img_height // 2:
        raise ValueError(
            f"embed_clinical_data_into_image: matrix_size ({matrix_size}) must be "
            f"less than half of the image dimensions ({img_height}x{img_width})."
        )

    if len(image_batch.shape) != 4:
        raise ValueError(
            "embed_clinical_data_into_image: Input must be a 4D tensor (B, C, H, W)."
        )

    xr_follow_up = tabular_batch[:, 0]  # followUpNumber (B,)
    age = tabular_batch[:, 1]  # patientAge (B,)
    gender = tabular_batch[:, 2]  # patientGender (B,)
    xr_position = tabular_batch[:, 3]  # viewPosition (B,)

    # Validate binary values using tensor operations
    if not torch.all((gender == 0) | (gender == 1)):
        invalid_gender = gender[(gender != 0) & (gender != 1)]
        raise ValueError(
            f"embed_clinical_data_into_image: Gender must be 0 or 1. "
            f"Found invalid values: {invalid_gender.tolist()}"
        )
    if not torch.all((xr_position == 0) | (xr_position == 1)):
        invalid_pos = xr_position[(xr_position != 0) & (xr_position != 1)]
        raise ValueError(
            f"embed_clinical_data_into_image: X-ray position must be 0 or 1. "
            f"Found invalid values: {invalid_pos.tolist()}"
        )

    # Create a copy of the image batch to avoid modifying the original
    # We can add .clone() if needed to avoid modifying the original tensor
    result = image_batch
    quad_size = matrix_size // 2

    # Prepare the clinical data for broadcasting
    # Reshape from (B,) to (B, 1, 1) for broadcasting to (B, quad_size, quad_size)
    xr_follow_up = xr_follow_up.view(-1, 1, 1).expand(-1, quad_size, quad_size)
    age = age.view(-1, 1, 1).expand(-1, quad_size, quad_size)
    gender = gender.view(-1, 1, 1).expand(-1, quad_size, quad_size)
    xr_position = xr_position.view(-1, 1, 1).expand(-1, quad_size, quad_size)

    # Embed the clinical data into all channels
    for c in range(result.shape[1]):  # Iterate over channels
        # Top-left quadrant: Follow-up number (normalized)
        result[:, c, :quad_size, :quad_size] = xr_follow_up
        # Top-right quadrant: Age (normalized)
        result[:, c, :quad_size, quad_size:matrix_size] = age
        # Bottom-left quadrant: Gender (0/1)
        result[:, c, quad_size:matrix_size, :quad_size] = gender
        # Bottom-right quadrant: View position (0/1)
        result[:, c, quad_size:matrix_size, quad_size:matrix_size] = xr_position

    return result
