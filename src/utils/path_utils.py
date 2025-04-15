"""Utility functions for path handling."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent.parent
