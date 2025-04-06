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


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

ENV_PRESENT = load_dotenv(PROJECT_ROOT / ".env")

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
OPTIMIZER = os.getenv("OPTIMIZER", "adam")
DEVICE = os.getenv("DEVICE", "cuda")


# Environment variables for logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_FORMAT = "[ %(levelname)-8s ] %(asctime)s | %(name)s | Message: %(message)s"


@dataclass
class Config:
    """
    Configuration class to hold all parameters.

    If environment variables are not set, default values are used.
    Attributes:
        num_workers (int): Number of workers for data loading.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        optimizer (str): Optimizer to use ('adam' or 'sgd').
        device (str): Device to use ('cuda' or 'cpu').
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (Path): Path to log file.
        log_format (str): Format string for log messages.

    Default values:
        num_workers: 4
        batch_size: 32
        num_epochs: 10
        learning_rate: 0.001
        optimizer: 'adam'
        device: 'cuda'
        log_level: 'INFO'
        log_file: PROJECT_ROOT / 'logs' / 'app.log'
        log_format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    Raises:
        ValueError: If any of the parameters are invalid.
        TypeError: If any of the parameters are of incorrect type.
    """

    num_workers: int = int(NUM_WORKERS) or 4
    batch_size: int = int(BATCH_SIZE) or 16
    num_epochs: int = int(NUM_EPOCHS) or 10
    learning_rate: float = float(LEARNING_RATE) or 0.001
    optimizer: Literal["sgd", "adam"] = OPTIMIZER or "adam"
    device: Literal["cuda", "cpu"] = DEVICE or "cuda"
    log_level: str = LOG_LEVEL or "info"
    log_file: Path = PROJECT_ROOT / "logs" / LOG_FILE
    log_format: str = (
        LOG_FORMAT or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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

    def __post_init__(self):
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

    def __repr__(self):
        return f"Config(num_workers={self.num_workers}, batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, optimizer={self.optimizer}, device={self.device})"
