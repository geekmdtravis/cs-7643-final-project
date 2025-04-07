"""Preprocessing functions for medical image datasets."""

import torch


def generate_image_lables(finding_labels: str) -> torch.Tensor:
    """
    Generate image labels from finding labels.

    Args:
        finding_labels (str): A string of finding labels separated by '|'.

    Returns:
        torch.Tensor: A tensor representing the image labels.
    """
    _fl = finding_labels.lower()

    if _fl.strip() == "":
        raise ValueError("Finding labels cannot be an empty string.")

    valid_labels = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "effusion",
        "emphysema",
        "fibrosis",
        "hernia",
        "infiltration",
        "mass",
        "no finding",
        "nodule",
        "pleural_thickening",
        "pneumonia",
        "pneumothorax",
    ]

    for label in _fl.split("|"):
        if label not in valid_labels:
            raise ValueError(f"Invalid finding label: {label}")

    image_labels = torch.zeros(15, dtype=torch.float32)
    image_labels[0] = 1 if "atelectasis" in _fl else 0
    image_labels[1] = 1 if "cardiomegaly" in _fl else 0
    image_labels[2] = 1 if "consolidation" in _fl else 0
    image_labels[3] = 1 if "edema" in _fl else 0
    image_labels[4] = 1 if "effusion" in _fl else 0
    image_labels[5] = 1 if "emphysema" in _fl else 0
    image_labels[6] = 1 if "fibrosis" in _fl else 0
    image_labels[7] = 1 if "hernia" in _fl else 0
    image_labels[8] = 1 if "infiltration" in _fl else 0
    image_labels[9] = 1 if "mass" in _fl else 0
    image_labels[10] = 1 if "no finding" in _fl else 0
    image_labels[11] = 1 if "nodule" in _fl else 0
    image_labels[12] = 1 if "pleural_thickening" in _fl else 0
    image_labels[13] = 1 if "pneumonia" in _fl else 0
    image_labels[14] = 1 if "pneumothorax" in _fl else 0

    return image_labels
