"""
Get system information
"""

import logging
import platform
import re
from dataclasses import dataclass

import psutil
import torch


@dataclass
class SystemInfo:
    """
    A class to represent system information.
    """

    os: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    pytorch_version: str
    device: str
    gpu_info: str
    gpu_memory: str

    def __str__(self):
        return (
            f"OS: {self.os}\n"
            f"PyTorch Version: {self.pytorch_version}\n"
            f"Device: {self.device.upper()}\n"
            f"CPU: {self.cpu_model}\n"
            f" - {self.cpu_cores} Cores (Physical)\n"
            f" - {self.cpu_threads} Threads (Logical)\n"
            f"GPU: {self.gpu_info}\n"
            f" - {self.gpu_memory} VRAM"
        )


def get_system_info() -> SystemInfo:
    """
    Get system information.

    Returns:
        SystemInfo: An instance of SystemInfo containing system details.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = "N/A"
    gpu_memory = "N/A"

    if torch.cuda.is_available():
        gpu_info = f"{torch.cuda.get_device_name(0)}"
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

    cpu_model = "Unknown"
    try:
        cpu_info = platform.processor()
        if cpu_info:
            cpu_model = cpu_info
        else:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = re.sub(".*model name.*:", "", line, 1).strip()
                        break
    except (OSError, RuntimeError) as e:
        logging.error("Error retrieving CPU information: %s", e)

    return SystemInfo(
        os=f"{platform.system()} ({platform.release()})",
        cpu_model=cpu_model,
        cpu_cores=psutil.cpu_count(logical=False),
        cpu_threads=psutil.cpu_count(logical=True),
        pytorch_version=torch.__version__,
        device=str(device),
        gpu_info=gpu_info,
        gpu_memory=gpu_memory,
    )


if __name__ == "__main__":
    system_info = get_system_info()
    print("\n=== System Information ===")
    print(system_info)
    print("\n===========================")
