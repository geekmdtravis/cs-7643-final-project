"""
A meta-model for the set of CXR multi-label classifier models
included in this repository, implemented as a wrapper.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import yaml

from src.models.densenet_121_multimodal import DenseNet121MultiModal
from src.models.densenet_121_vanilla import DenseNet121Vanilla
from src.models.densenet_201_multimodal import DenseNet201MultiModal
from src.models.densenet_201_vanilla import DenseNet201Vanilla
from src.models.vit_b_16_multimodal import ViTB16MultiModal
from src.models.vit_b_16_vanilla import ViTB16Vanilla
from src.models.vit_b_32_multimodal import ViTB32MultiModal
from src.models.vit_b_32_vanilla import ViTB32Vanilla
from src.models.vit_l_16_multimodal import ViTL16MultiModal
from src.models.vit_l_16_vanilla import ViTL16Vanilla

SupportedModels = Literal[
    "densenet121",
    "densenet121_mm",
    "densenet201",
    "densenet201_mm",
    "vit_b_16",
    "vit_b_16_mm",
    "vit_b_32",
    "vit_b_32_mm",
    "vit_l_16",
    "vit_l_16_mm",
]


@dataclass
class CXRModelConfig:
    """
    Configuration class for the CXR model.

    Attributes:
        model (SupportedModels): The name of the model to use.
            Supported models are:
            - densenet121
            - densenet121_mm
            - densenet201
            - densenet201_mm
            - vit_b_16
            - vit_b_16_mm
            - vit_b_32
            - vit_b_32_mm
            - vit_l_16
            - vit_l_16_mm
        hidden_dims (tuple[int] | list[int] | None): Hidden dimensions for
            the classifier. Defaults to None.
        dropout (float): Dropout rate for the classifier.
        num_classes (int): Number of output classes. Defaults to 15
            (14 pathologies + 1 no pathology).
        tabular_features (int): Number of tabular features to combine with image
            features. Defaults to 4 due to four clinical features being
            present in the dataset.
        freeze_backbone (bool): Whether to freeze the backbone model parameters
            during training. Defaults to False. When set to True will freeze
            all parameters in the ViT-L/16 model except for the classifier head.
    """

    model: SupportedModels
    hidden_dims: tuple[int] | list[int] | None = None
    dropout: float = 0.2
    num_classes: int = 15
    tabular_features: int = 4
    freeze_backbone: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "CXRModelConfig":
        """
        Load the configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file. This path
                should be relative to the root of the repository.
                Example: "configs/cxr_model_config.yaml"

        Returns:
            CXRModelConfig: An instance of CXRModelConfig with the loaded
                configuration.

        Example:
            config = CXRModelConfig.from_yaml("config.yaml")
        """

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Convert hidden_dims list to tuple if present
        if "hidden_dims" in config and isinstance(config["hidden_dims"], list):
            config["hidden_dims"] = tuple(config["hidden_dims"])

        return cls(**config)

    def as_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return {
            "model": self.model,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "tabular_features": self.tabular_features,
            "freeze_backbone": self.freeze_backbone,
        }

    def __repr__(self) -> str:
        """
        String representation of the configuration.

        Returns:
            str: A string representation of the configuration.
        """
        return f"CXRModelConfig({self.as_dict()})"

    def __str__(self) -> str:
        """
        String representation of the configuration.

        Returns:
            str: A string representation of the configuration.
        """
        return f"CXRModelConfig({self.as_dict()})"


class CXRModel(nn.Module):
    """
    A meta-model for the set of CXR multi-label classifier models
    included in this repository, implemented as a wrapper.
    """

    def __init__(
        self,
        model: SupportedModels,
        hidden_dims: tuple[int] | list[int] | None = None,
        dropout: float = 0.2,
        num_classes: int = 15,
        tabular_features: int = 4,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the CXR Model.
        Args:
            hidden_dims (tuple[int] | list[int] | None): Hidden dimensions for
                the classifier. Defaults to None. When None is provided,
                the model will not use hidden layers and the default
                classification head will be used, where the output from
                the backbone is passed directly to the classifier.
            dropout (float): Dropout rate for the classifier
            num_classes (int): Number of output classes. Defaults to 15
                (14 pathologies + 1 no pathology)
            tabular_features (int): Number of tabular features to combine with image
                features. Defaults to 4 due to four clinical features being
                present in the dataset
            freeze_backbone (bool): Whether to freeze the backbone model parameters
                during training. Defaults to False. When set to True will freeze
                all parameters in the ViT-L/16 model except for the classifier
                head.

            NOTE: Demo mode is not supported in this wrapper class. If you want to
            use demo mode, you will need to create an instance of the model
            class directly.
        """
        super(CXRModel, self).__init__()

        hidden_dims = hidden_dims if hidden_dims is not None else ()

        self.model_name = model

        if model not in [
            "densenet121",
            "densenet121_mm",
            "densenet201",
            "densenet201_mm",
            "vit_b_16",
            "vit_b_16_mm",
            "vit_b_32",
            "vit_b_32_mm",
            "vit_l_16",
            "vit_l_16_mm",
        ]:
            raise ValueError(f"Model {model} is not supported.")
        if model == "densenet121":
            self.model = DenseNet121Vanilla(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif model == "densenet121_mm":
            self.model = DenseNet121MultiModal(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                tabular_features=tabular_features,
                freeze_backbone=freeze_backbone,
            )
        elif model == "densenet201":
            self.model = DenseNet201Vanilla(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif model == "densenet201_mm":
            self.model = DenseNet201MultiModal(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                tabular_features=tabular_features,
                freeze_backbone=freeze_backbone,
            )
        elif model == "vit_b_16":
            self.model = ViTB16Vanilla(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif model == "vit_b_16_mm":
            self.model = ViTB16MultiModal(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                tabular_features=tabular_features,
                freeze_backbone=freeze_backbone,
            )
        elif model == "vit_b_32":
            self.model = ViTB32Vanilla(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif model == "vit_b_32_mm":
            self.model = ViTB32MultiModal(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                tabular_features=tabular_features,
                freeze_backbone=freeze_backbone,
            )
        elif model == "vit_l_16":
            self.model = ViTL16Vanilla(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif model == "vit_l_16_mm":
            self.model = ViTL16MultiModal(
                hidden_dims=hidden_dims,
                dropout=dropout,
                num_classes=num_classes,
                tabular_features=tabular_features,
                freeze_backbone=freeze_backbone,
            )
        else:
            raise ValueError(f"Model {model} is not supported.")

    def forward(
        self, img_batch: torch.Tensor, tabular_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model. Of note, while
        the tabular data is passed to the model, it is not used in the
        forward pass of the vanilla models. It is only used in the
        forward pass of the multi-modal models. It's inclusion is
        an artifact of that, and offers no performance detriment since
        the DataLoader always returns tabular data.

        Args:
            img_batch (torch.Tensor): Batch of images to be passed to the model.
            tabular_batch (torch.Tensor): Batch of tabular data to be passed to the
                model. This is only used in the multi-modal models.
        Returns:
            torch.Tensor: Output tensor from the model.
        """
        if self.model_name in [
            "densenet121",
            "densenet201",
            "vit_b_16",
            "vit_b_32",
            "vit_l_16",
        ]:
            return self.model(img_batch)
        elif self.model_name in [
            "densenet121_mm",
            "densenet201_mm",
            "vit_b_16_mm",
            "vit_b_32_mm",
            "vit_l_16_mm",
        ]:
            return self.model(img_batch, tabular_batch)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
