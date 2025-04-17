"""This script is used to run the main functionality of the application."""

from src.utils import get_system_info

SEED = 42


def print_intro():
    """Prints the introduction of the application."""
    print("===============================================")
    print("CS7643 Final Project - Multimodal Chest X-ray Classification")
    print("-----------------------------------------------")
    print(f"System Information:\n{get_system_info()}")
    print("-----------------------------------------------")


if __name__ == "__main__":
    print_intro()
