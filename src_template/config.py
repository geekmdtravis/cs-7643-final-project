from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


@dataclass
class DataConfig:
    # Data paths (Update these based on where kagglehub downloads or where you move the data)
    base_data_path: Path = Path(
        "~/.cache/kagglehub/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized"
    ).expanduser()
    image_dir: Path = base_data_path / "images-224"
    metadata_path: Path = base_data_path / "Data_Entry_2017.csv"
    train_val_list_path: Path = base_data_path / "train_val_list_NIH.txt"
    test_list_path: Path = base_data_path / "test_list_NIH.txt"

    # Image parameters
    image_size: tuple[int, int] = (224, 224)  # Images are already 224x224
    num_channels: int = 3  # 3 for RGB (duplicate grayscale), 1 for grayscale

    # Data loading
    batch_size: int = 32
    num_workers: int = 4

    # Tabular data fields
    tabular_features = ["Patient Age", "Patient Gender", "View Position"]

    # Integration method configuration
    class EmbeddingConfig:
        # For Method 1: Image embedding of tabular data
        box_size: int = 20  # Size of each metadata box in pixels
        margin: int = 2  # Margin between boxes
        position: Literal["top", "bottom", "right"] = (
            "right"  # Where to add metadata boxes
        )

    class FusionConfig:
        # For Method 2: Feature/token fusion
        embedding_dim: int = 64  # Dimension for tabular feature embedding
        dropout: float = 0.1


@dataclass
class ModelConfig:
    # Model type selection
    model_type: Literal[
        "cnn_baseline",
        "vit_baseline",
        "cnn_embed",
        "cnn_fusion",
        "vit_embed",
        "vit_fusion",
    ]

    # CNN specific parameters
    class CNNConfig:
        backbone: str = "resnet50"  # Base CNN architecture
        pretrained: bool = True
        freeze_backbone: bool = False

    # ViT specific parameters
    class ViTConfig:
        model_name: str = "vit_base_patch16_224"  # Base ViT architecture
        pretrained: bool = True
        patch_size: int = 16
        num_heads: int = 12
        mlp_dim: int = 3072
        dropout: float = 0.1


@dataclass
class TrainingConfig:
    # Basic training parameters
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler_type: Optional[str] = "cosine"
    warmup_epochs: int = 5

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Mixed precision training
    use_amp: bool = True

    # Checkpoint configuration
    save_freq: int = 5  # Save every N epochs
    keep_top_k: int = 3  # Keep top K best models


@dataclass
class Config:
    # Experiment name for logging
    experiment_name: str

    # Random seed for reproducibility
    seed: int = 42

    # Device configuration
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Sub-configurations
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig(
        model_type="cnn_baseline"
    )  # Default to CNN baseline
    training: TrainingConfig = TrainingConfig()

    # Directories
    output_dir: Path = Path("results")

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.experiment_name

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs" / self.experiment_name

    @property
    def plot_dir(self) -> Path:
        return self.output_dir / "plots" / self.experiment_name

    def create_directories(self):
        """Create all necessary directories for the experiment."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
