import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.cxr_model import CXRModel, CXRModelConfig


class TestCXRModelConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.yaml"

        # Write test config
        self.config_path.write_text(
            """
model: vit_b_16
hidden_dims: [256, 128]
dropout: 0.5
num_classes: 10
tabular_features: 8
freeze_backbone: true
"""
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)

    def test_default_initialization(self):
        """Test CXRModelConfig initialization with default values."""
        config = CXRModelConfig(model="densenet121")
        self.assertEqual(config.model, "densenet121")
        self.assertEqual(config.hidden_dims, None)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.num_classes, 15)
        self.assertEqual(config.tabular_features, 4)
        self.assertEqual(config.freeze_backbone, False)

    def test_custom_initialization(self):
        """Test CXRModelConfig initialization with custom values."""
        config = CXRModelConfig(
            model="vit_b_16",
            hidden_dims=(256, 128),
            dropout=0.5,
            num_classes=10,
            tabular_features=8,
            freeze_backbone=True,
        )
        self.assertEqual(config.model, "vit_b_16")
        self.assertEqual(config.hidden_dims, (256, 128))
        self.assertEqual(config.dropout, 0.5)
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.tabular_features, 8)
        self.assertEqual(config.freeze_backbone, True)

    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        config = CXRModelConfig.from_yaml(str(self.config_path))
        self.assertEqual(config.model, "vit_b_16")
        self.assertEqual(config.hidden_dims, (256, 128))
        self.assertEqual(config.dropout, 0.5)
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.tabular_features, 8)
        self.assertTrue(config.freeze_backbone)


class TestCXRModel(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.batch_size = 2
        self.img_channels = 3
        self.img_size = 224
        self.tabular_features = 4
        self.num_classes = 15

        # Create dummy input data
        self.img_batch = torch.randn(
            self.batch_size, self.img_channels, self.img_size, self.img_size
        )
        self.tabular_batch = torch.randn(self.batch_size, self.tabular_features)

    def test_init_vanilla_models(self):
        """Test initialization of vanilla models."""
        vanilla_models = [
            "densenet121",
            "densenet201",
            "vit_b_16",
            "vit_b_32",
            "vit_l_16",
        ]
        for model_name in vanilla_models:
            model = CXRModel(model=model_name)
            self.assertIsInstance(model, CXRModel)
            self.assertEqual(model.model_name, model_name)

    def test_init_multimodal_models(self):
        """Test initialization of multimodal models."""
        mm_models = [
            "densenet121_mm",
            "densenet201_mm",
            "vit_b_16_mm",
            "vit_b_32_mm",
            "vit_l_16_mm",
        ]
        for model_name in mm_models:
            model = CXRModel(model=model_name)
            self.assertIsInstance(model, CXRModel)
            self.assertEqual(model.model_name, model_name)

    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with self.assertRaises(ValueError):
            CXRModel(model="invalid_model")

    def test_forward_vanilla_models(self):
        """Test forward pass for vanilla models."""
        models = ["densenet121", "vit_b_16", "densenet201", "vit_b_32", "vit_l_16"]

        for model_name in models:
            model = CXRModel(model=model_name)
            output = model(self.img_batch, self.tabular_batch)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_forward_multimodal_models(self):
        """Test forward pass for multimodal models."""
        models = [
            "densenet121_mm",
            "vit_b_16_mm",
            "densenet201_mm",
            "vit_b_32_mm",
            "vit_l_16_mm",
        ]
        for model_name in models:
            model = CXRModel(model=model_name)
            output = model(self.img_batch, self.tabular_batch)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_custom_dimensions(self):
        """Test model with custom hidden dimensions."""
        custom_dims = (256, 128, 64)
        model = CXRModel(model="densenet121", hidden_dims=custom_dims, dropout=0.3)
        output = model(self.img_batch, self.tabular_batch)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        model = CXRModel(model="densenet121", freeze_backbone=True)

        # Check that backbone parameters are frozen
        for name, param in model.model.named_parameters():
            if "classifier" not in name:  # backbone parameters
                self.assertFalse(param.requires_grad)
            else:  # classifier parameters
                self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
