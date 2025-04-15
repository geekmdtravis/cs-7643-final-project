"""
Images downloaded via Kaggle Hub will need to be
prepared for local use. There are two forms each image
must take: (1) The original, unmodified image, and
(2) a version with clinical tabular data embedded as
a matrix into the upper-left corner of the image.
This script will copy the origimal images to
{workspace}/artifacts/original_images and the embedded
images to {workspace}/artifacts/embedded_images.
"""

from pathlib import Path

# from PIL import Image
from tqdm import tqdm

# from src.utils import embed_clinical_data_into_image
from src.utils.path_utils import get_project_root


def copy_images_to_artifacts(
    origin_dir: Path,
    dest_dir: Path | None = None,
) -> None:
    """
    The origin dir is the images folder within the Kaggle
    cached dataset. The destination dir will default to
    {workspace}/artifacts/cxr_images.
    """
    if dest_dir is None:
        project_root = get_project_root()
        artifacts_dir = project_root / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir.mkdir(parents=True)
        dest_dir = artifacts_dir / "cxr_images"

    if not origin_dir.exists():
        raise FileNotFoundError(f"{origin_dir} does not exist.")
    if not origin_dir.is_dir():
        raise NotADirectoryError(f"{origin_dir} is not a directory.")
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"{dest_dir} is not a directory.")

    image_list = list(origin_dir.glob("*.png"))
    for image in tqdm(image_list, desc="Copying images"):
        if image.is_file():
            dest_image = dest_dir / image.name
            dest_image.write_bytes(image.read_bytes())
        else:
            raise FileNotFoundError(f"{image} is not a file.")
    print(f"Copied images from {origin_dir} to {dest_dir}.")


def copy_csv_to_artifacts(
    origin_file: Path,
    dest_file: Path | None = None,
) -> None:
    """
    The origin file is the csv within the Kaggle
    cached dataset. The destination file will default to
    {workspace}/artifacts/tabular_data.csv.
    """
    if dest_file is None:
        project_root = get_project_root()
        artifacts_dir = project_root / "artifacts"
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"{artifacts_dir} does not exist.")
        dest_dir = artifacts_dir
        dest_file = dest_dir / "clinical_data.csv"

    if not origin_file.exists():
        raise FileNotFoundError(f"{origin_file} does not exist.")
    if not origin_file.is_file():
        raise FileNotFoundError(f"{origin_file} is not a file or was not found.")
    if not dest_dir.exists():
        dest_dir.mkdir()
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"{dest_dir} is not a directory.")

    dest_file.write_bytes(origin_file.read_bytes())
    print(f"Copied csv from {origin_file} to {dest_file}.")


# def copy_embedded_images_to_artifacts(
#     origin_dir: Path,
#     dest_dir: Path | None = None,
# ) -> None:
#     """
#     The origin dir is the images folder within the Kaggle
#     cached dataset. The destination dir will default to
#     {workspace}/artifacts/embedded_images.
#     """
#     if dest_dir is None:
#         project_root = get_project_root()
#         artifacts_dir = project_root / "artifacts"
#         if not artifacts_dir.exists():
#             artifacts_dir.mkdir(parents=True)
#         dest_dir = artifacts_dir / "embedded_images"

#     if not origin_dir.exists():
#         raise FileNotFoundError(f"{origin_dir} does not exist.")
#     if not origin_dir.is_dir():
#         raise NotADirectoryError(f"{origin_dir} is not a directory.")
#     if not dest_dir.exists():
#         dest_dir.mkdir(parents=True)
#     if not dest_dir.is_dir():
#         raise NotADirectoryError(f"{dest_dir} is not a directory.")

#     image_list = list(origin_dir.glob("*.png"))
#     for image in tqdm(image_list, desc="Copying embedded images"):
#         if image.is_file():
#             print("Not implemented yet")
