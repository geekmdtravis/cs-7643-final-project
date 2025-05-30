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

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PRESENT = load_dotenv(PROJECT_ROOT / ".env")

DEVICE = os.getenv("DEVICE", "cuda")
SEED = int(os.getenv("SEED", "42"))
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
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

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
        class_labels (tuple[str]): List of class labels for the dataset.
        cxr_test_dir: Path to unmodified CXR images for testing.
        cxr_train_dir: Path to unmodified CXR images for training.
        demo_dir (Path): Path to the demo directory.
        device (str): Device to use ('cuda' or 'cpu').
        embedded_test_dir: Path to embedded CXR images for testing.
        embedded_train_dir: Path to embedded CXR images for training.
        log_file (Path): Path to log file.
        log_format (str): Format string for log messages.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        project_root (Path): Path to the project root directory.
        seed (int): Random seed for reproducibility.
        tabular_clinical_test (Path): Path to testing clinical
            tabular CSV data.
        tabular_clinical_train (Path): Path to training clinical
            tabular CSV data.

    Default values:
        artifacts_dir: PROJECT_ROOT / 'artifacts'
        cxr_test_dir: PROJECT_ROOT / 'artifacts' / 'cxr_test'
        cxr_train_dir: PROJECT_ROOT / 'artifacts' / 'cxr_train'
        demo_dir: PROJECT_ROOT / 'artifacts' / 'demo'
        device: 'cuda'
        embedded_test_dir: PROJECT_ROOT / 'artifacts' / 'embedded_test'
        embedded_train_dir: PROJECT_ROOT / 'artifacts' / 'embedded_train'
        log_file: PROJECT_ROOT / 'logs' / 'app.log'
        log_format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_level: 'INFO'
        project_root: Path to the project root directory.
        results_dir: PROJECT_ROOT / 'results'
        seed: 42
        tabular_clinical_test: PROJECT_ROOT / 'artifacts' / 'test.csv'
        tabular_clinical_train: PROJECT_ROOT / 'artifacts' / 'train.csv'


    Raises:
        ValueError: If any of the parameters are invalid.
        TypeError: If any of the parameters are of incorrect type.
    """

    results_dir: Path = PROJECT_ROOT / "results"
    artifacts: Path = PROJECT_ROOT / "artifacts"
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
    embedded32_test_dir: Path = PROJECT_ROOT / "artifacts" / "embedded32_test"
    embedded32_train_dir: Path = PROJECT_ROOT / "artifacts" / "embedded32_train"
    embedded32_val_dir: Path = PROJECT_ROOT / "artifacts" / "embedded32_val"
    log_file: Path = PROJECT_ROOT / "logs" / LOG_FILE
    log_format: str = (
        LOG_FORMAT or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_level: str = LOG_LEVEL or "info"
    project_root: Path = PROJECT_ROOT
    seed: int = int(SEED) or 42
    tabular_clinical_test: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    tabular_clinical_train: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    tabular_clinical_val: Path = PROJECT_ROOT / "artifacts" / "val.csv"

    def __init__(self):

        setup_logging(
            log_level=self.log_level, log_file=self.log_file, log_format=self.log_format
        )

        if not ENV_PRESENT:
            logging.info("Environment variables not found. Using default values.")
            logging.warning(".env file not found. Using default values.")
        else:
            logging.info("Environment variables loaded successfully.")
            logging.debug("Config: %s", self)

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        if not isinstance(self.log_format, str):
            raise TypeError("log_format must be a string")

        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be either 'cuda' or 'cpu'")
        if not isinstance(self.device, str):
            raise TypeError("device must be a 'cpu' or 'cuda'")
        if not self.artifacts.exists():
            logging.info("Config: Artifacts directory does not exist. Creating it.")
            self.artifacts.mkdir(parents=True, exist_ok=True)
        if not self.demo_dir.exists():
            logging.info("Config: Demo directory does not exist. Creating it.")
            self.demo_dir.mkdir(parents=True, exist_ok=True)
        if not self.embedded32_test_dir.exists():
            logging.info(
                "Config: Embedded32 test directory does not exist. Creating it."
            )
            self.embedded32_test_dir.mkdir(parents=True, exist_ok=True)
        if not self.embedded32_train_dir.exists():
            logging.info(
                "Config: Embedded32 train directory does not exist. Creating it."
            )
            self.embedded32_train_dir.mkdir(parents=True, exist_ok=True)
        if not self.embedded32_val_dir.exists():
            logging.info(
                "Config: Embedded32 val directory does not exist. Creating it."
            )
            self.embedded32_val_dir.mkdir(parents=True, exist_ok=True)
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
