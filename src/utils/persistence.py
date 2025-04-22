"""
Utility functions to simplify the process of saving and
loading the CXRModel to and from disk.
"""

from pathlib import Path

import torch

from src.models import CXRModel, CXRModelConfig


def save_model(model: CXRModel, config: CXRModelConfig, file_path: str) -> Path:
    """
    Save the model to disk.

    Args:
        model (CXRModel): The model to save.
        config (CXRModelConfig): The configuration of the model.
        file_path (str): The path to save the model to. If the
            path does not exist, it will be created. If the
            'pth' extension is not present, it will be added.
    Returns:
        a Path object representing the path to the saved model.
    """
    path = Path(file_path)

    if not path.suffix == ".pth":
        path = path.with_suffix(".pth")

    save_info = {
        "model": model.state_dict(),
        "config": config.as_dict(),
    }

    torch.save(save_info, path)

    return path


def load_model(file_path: str) -> CXRModel:
    """
    Load the model from disk.

    Args:
        file_path (str): The path to load the model from. If the
            path does not exist, an error will be raised.
    Returns:
        CXRModel: The loaded model.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Path {path} is not a file.")
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

    save_info = torch.load(path)

    model = CXRModel(**save_info["config"])
    model.load_state_dict(save_info["model"])

    return model
