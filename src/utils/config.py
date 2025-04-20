"""
Configuration module for the project.
This module loads configuration parameters from environment variables
and provides a dataclass to hold these parameters.
"""

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

ENV_PRESENT = load_dotenv(PROJECT_ROOT / ".env")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEVICE = os.getenv("DEVICE", "cuda")
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
OPTIMIZER = os.getenv("OPTIMIZER", "adam")
SEED = int(os.getenv("SEED", "42"))


# Environment variables for logging
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

LOG_FORMAT = "[ %(levelname)-8s ] %(asctime)s | %(name)s | Message: %(message)s"


def setup_logging(
    log_level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    log_file: Path | str = "app.log",
    log_format: str = "[ %(levelname)s ] %(asctime)s | %(name)s | Message: %(message)s",
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure logging with both file and console handlers.

    Args:
        log_level: Minimum logging level (debug, info, warning, error, critical)
        log_file: Path to log file. If None, only console logging is enabled
        log_format: Format string for log messages
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


@dataclass
class Config:
    """
    Configuration class to hold all parameters.

    If environment variables are not set, default values are used.
    Attributes:
        artifacts_dir (Path): Path to the artifacts directory.
        batch_size (int): Batch size for training.
        tabular_clinical_train (Path): Path to training clinical
            tabular CSV data.
        tabular_clinical_test (Path): Path to testing clinical
            tabular CSV data.
        class_labels (tuple[str]): List of class labels for the dataset.
        cxr_test_dir: Path to unmodified CXR images for testing.
        cxr_train_dir: Path to unmodified CXR images for training.
        embedded_test_dir: Path to embedded CXR images for testing.
        embedded_train_dir: Path to embedded CXR images for training.
        demo_dir (Path): Path to the demo directory.
        device (str): Device to use ('cuda' or 'cpu').
        learning_rate (float): Learning rate for the optimizer.
        log_file (Path): Path to log file.
        log_format (str): Format string for log messages.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        num_epochs (int): Number of epochs for training.
        num_workers (int): Number of workers for data loading.
        optimizer (str): Optimizer to use ('adam' or 'sgd').
        project_root (Path): Path to the project root directory.
        seed (int): Random seed for reproducibility.

    Default values:
        artifacts_dir: PROJECT_ROOT / 'artifacts'
        batch_size: 32
        tabular_clinical_train: PROJECT_ROOT / 'artifacts' / 'train.csv'
        tabular_clinical_test: PROJECT_ROOT / 'artifacts' / 'test.csv'
        cxr_test_dir: PROJECT_ROOT / 'artifacts' / 'cxr_test'
        cxr_train_dir: PROJECT_ROOT / 'artifacts' / 'cxr_train'
        demo_dir: PROJECT_ROOT / 'artifacts' / 'demo'
        device: 'cuda'
        embedded_test_dir: PROJECT_ROOT / 'artifacts' / 'embedded_test'
        embedded_train_dir: PROJECT_ROOT / 'artifacts' / 'embedded_train'
        learning_rate: 0.001
        log_file: PROJECT_ROOT / 'logs' / 'app.log'
        log_format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_level: 'INFO'
        num_epochs: 10
        num_workers: 4
        optimizer: 'adam'
        project_root: Path to the project root directory.
        seed: 42


    Raises:
        ValueError: If any of the parameters are invalid.
        TypeError: If any of the parameters are of incorrect type.
    """

    results_dir: Path = PROJECT_ROOT / "results"
    artifacts: Path = PROJECT_ROOT / "artifacts"
    batch_size: int = int(BATCH_SIZE) or 16
    class_labels: tuple[str] = (
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural Thickening",
        "Pneumonia",
        "Pneumothorax",
    )
    cxr_test_dir: Path = PROJECT_ROOT / "artifacts" / "cxr_test"
    cxr_train_dir: Path = PROJECT_ROOT / "artifacts" / "cxr_train"
    cxr_val_dir = PROJECT_ROOT / "artifacts" / "cxr_val"
    demo_dir: Path = PROJECT_ROOT / "artifacts" / "demo"
    device: Literal["cuda", "cpu"] = DEVICE or "cuda"
    embedded_test_dir: Path = PROJECT_ROOT / "artifacts" / "embedded_test"
    embedded_train_dir: Path = PROJECT_ROOT / "artifacts" / "embedded_train"
    embedded_val_dir: Path = PROJECT_ROOT / "artifacts" / "embedded_val"
    learning_rate: float = float(LEARNING_RATE) or 0.001
    log_file: Path = PROJECT_ROOT / "logs" / LOG_FILE
    log_format: str = (
        LOG_FORMAT or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_level: str = LOG_LEVEL or "info"
    num_epochs: int = int(NUM_EPOCHS) or 10
    num_workers: int = int(NUM_WORKERS) or 4
    optimizer: Literal["sgd", "adam"] = OPTIMIZER or "adam"
    project_root: Path = PROJECT_ROOT
    seed: int = int(SEED) or 42
    tabular_clinical_test: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    tabular_clinical_train: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    tabular_clinical_val: Path = PROJECT_ROOT / "artifacts" / "val.csv"

    def __init__(self):

        # Set up logging
        setup_logging(
            log_level=self.log_level, log_file=self.log_file, log_format=self.log_format
        )

        if not ENV_PRESENT:
            logging.info("Environment variables not found. Using default values.")
            logging.warning(".env file not found. Using default values.")
        else:
            logging.info("Environment variables loaded successfully.")
            logging.debug("Config: %s", self)

        # Validate logging parameters
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        if not isinstance(self.log_format, str):
            raise TypeError("log_format must be a string")

        # Validate other parameters
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.optimizer not in ["adam", "sgd"]:
            raise ValueError("optimizer must be either 'adam' or 'sgd'")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be either 'cuda' or 'cpu'")
        if not isinstance(self.device, str):
            raise TypeError("device must be a 'cpu' or 'cuda'")
        if not isinstance(self.optimizer, str):
            raise TypeError("optimizer must be 'sgd' or 'adam'")
        if not isinstance(self.num_workers, int):
            raise TypeError("num_workers must be an integer")
        if not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if not isinstance(self.num_epochs, int):
            raise TypeError("num_epochs must be an integer")
        if not isinstance(self.learning_rate, float):
            raise TypeError("learning_rate must be a float")
        if not self.artifacts.exists():
            logging.info("Config: Artifacts directory does not exist. Creating it.")
            self.artifacts.mkdir(parents=True, exist_ok=True)
        if not self.demo_dir.exists():
            logging.info("Config: Demo directory does not exist. Creating it.")
            self.demo_dir.mkdir(parents=True, exist_ok=True)
        if not self.embedded_train_dir.exists():
            logging.info(
                "Config: Embedded train directory does not exist. Creating it."
            )
            self.embedded_train_dir.mkdir(parents=True, exist_ok=True)
        if not self.embedded_test_dir.exists():
            logging.info("Config: Embedded test directory does not exist. Creating it.")
            self.embedded_test_dir.mkdir(parents=True, exist_ok=True)
        if not self.cxr_train_dir.exists():
            logging.info("Config: CXR train directory does not exist. Creating it.")
            self.cxr_train_dir.mkdir(parents=True, exist_ok=True)
        if not self.cxr_test_dir.exists():
            logging.info("Config: CXR test directory does not exist. Creating it.")
            self.cxr_test_dir.mkdir(parents=True, exist_ok=True)
        if not self.cxr_val_dir.exists():
            logging.info("Config: CXR val directory does not exist. Creating it.")
            self.cxr_val_dir.mkdir(parents=True, exist_ok=True)
        if not self.embedded_val_dir.exists():
            logging.info("Config: Embedded val directory does not exist. Creating it.")
            self.embedded_val_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (
            f"Config(num_workers={self.num_workers}, "
            f"batch_size={self.batch_size}, "
            f"num_epochs={self.num_epochs}, "
            f"learning_rate={self.learning_rate}, "
            f"optimizer={self.optimizer}, device={self.device})"
        )
