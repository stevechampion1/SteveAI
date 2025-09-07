# SteveAI - Utility Functions

import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        log_file: Optional log file path
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )

    logger.info(f"Logging configured with level: {level}")


def print_memory_usage(step_name: str = "") -> None:
    """
    Print current memory usage.

    Args:
        step_name: Name of the step for logging
    """
    try:
        # CPU memory
        process = psutil.Process(os.getpid())
        ram_used_gb = process.memory_info().rss / 1024**3
        ram_total_gb = psutil.virtual_memory().total / 1024**3
        ram_percent = psutil.virtual_memory().percent

        logger.info(
            f"[{step_name}] RAM Usage: {ram_percent:.1f}% ({ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB)"
        )

        # GPU memory
        if torch.cuda.is_available():
            try:
                gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                device_index = torch.cuda.current_device()
                gpu_mem_total = (
                    torch.cuda.get_device_properties(device_index).total_memory
                    / 1024**3
                )

                logger.info(
                    f"[{step_name}] GPU-{device_index} VRAM: "
                    f"Allocated={gpu_mem_alloc:.2f} GB, "
                    f"Reserved={gpu_mem_reserved:.2f} GB, "
                    f"Total={gpu_mem_total:.2f} GB"
                )
            except Exception as e:
                logger.warning(f"Could not get detailed GPU memory info: {e}")
    except Exception as e:
        logger.warning(f"Could not get memory usage info: {e}")


def check_disk_space(path: str = ".") -> float:
    """
    Check available disk space.

    Args:
        path: Path to check disk space for

    Returns:
        Available disk space in GB
    """
    try:
        total, used, free = shutil.disk_usage(path)
        free_space_gb = free / 1024**3
        logger.info(f"Free disk space in '{path}': {free_space_gb:.2f} GB")
        return free_space_gb
    except FileNotFoundError:
        logger.warning(f"Path not found for disk usage check: {path}")
        return 0.0


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.

    Returns:
        Dictionary containing system information
    """
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        "disk_total_gb": shutil.disk_usage(".").total / 1024**3,
        "disk_free_gb": shutil.disk_usage(".").free / 1024**3,
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        "platform": psutil.sys.platform,
    }

    # GPU information
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_info"] = []
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory
                / 1024**3,
                "compute_capability": torch.cuda.get_device_properties(i).major,
            }
            info["gpu_info"].append(gpu_info)
    else:
        info["gpu_count"] = 0
        info["gpu_info"] = []

    return info


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Data saved to JSON file: {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Data loaded from JSON file: {file_path}")
    return data


def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data to pickle file.

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Data saved to pickle file: {file_path}")


def load_pickle(file_path: str) -> Any:
    """
    Load data from pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Loaded data
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    logger.info(f"Data loaded from pickle file: {file_path}")
    return data


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_directory_size(directory_path: str) -> int:
    """
    Get total size of directory in bytes.

    Args:
        directory_path: Path to the directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def clean_directory(directory_path: str, pattern: str = "*") -> None:
    """
    Clean directory by removing files matching pattern.

    Args:
        directory_path: Path to the directory
        pattern: File pattern to match
    """
    if not os.path.exists(directory_path):
        return

    removed_count = 0
    for file_path in Path(directory_path).glob(pattern):
        if file_path.is_file():
            file_path.unlink()
            removed_count += 1

    logger.info(f"Removed {removed_count} files from {directory_path}")


def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def copy_file(source: str, destination: str) -> None:
    """
    Copy file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    logger.info(f"File copied from {source} to {destination}")


def move_file(source: str, destination: str) -> None:
    """
    Move file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.move(source, destination)
    logger.info(f"File moved from {source} to {destination}")


def get_timestamp() -> str:
    """
    Get current timestamp string.

    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_backup(file_path: str, backup_dir: Optional[str] = None) -> str:
    """
    Create backup of a file.

    Args:
        file_path: Path to the file to backup
        backup_dir: Directory to store backup (default: same directory)

    Returns:
        Path to the backup file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if backup_dir is None:
        backup_dir = os.path.dirname(file_path)

    ensure_directory(backup_dir)

    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    timestamp = get_timestamp()
    backup_filename = f"{name}_backup_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)

    copy_file(file_path, backup_path)
    return backup_path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max.

    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def calculate_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> float:
    """
    Calculate accuracy between predictions and targets.

    Args:
        predictions: Predicted tokens
        targets: Target tokens
        ignore_index: Index to ignore in calculation

    Returns:
        Accuracy as float between 0 and 1
    """
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0

    correct = (predictions == targets) & mask
    return correct.sum().float() / mask.sum().float()


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to: {seed}")


def log_system_info() -> None:
    """Log system information."""
    info = get_system_info()

    logger.info("=== System Information ===")
    logger.info(f"CPU Count: {info['cpu_count']}")
    logger.info(f"Memory Total: {info['memory_total_gb']:.2f} GB")
    logger.info(f"Memory Available: {info['memory_available_gb']:.2f} GB")
    logger.info(f"Disk Total: {info['disk_total_gb']:.2f} GB")
    logger.info(f"Disk Free: {info['disk_free_gb']:.2f} GB")
    logger.info(f"Python Version: {info['python_version']}")
    logger.info(f"Platform: {info['platform']}")

    if info["gpu_count"] > 0:
        logger.info(f"GPU Count: {info['gpu_count']}")
        for gpu in info["gpu_info"]:
            logger.info(
                f"  GPU {gpu['device_id']}: {gpu['name']} "
                f"({gpu['memory_total_gb']:.2f} GB)"
            )
    else:
        logger.info("No GPU available")


# Example usage and testing functions
def test_utils():
    """Test utility functions."""
    logger.info("Testing utility functions...")

    # Test memory usage
    print_memory_usage("Test")

    # Test disk space
    free_space = check_disk_space()
    logger.info(f"Free disk space: {free_space:.2f} GB")

    # Test system info
    system_info = get_system_info()
    logger.info(f"System info keys: {list(system_info.keys())}")

    # Test time formatting
    time_str = format_time(3661.5)
    logger.info(f"Formatted time: {time_str}")

    # Test size formatting
    size_str = format_size(1024**3 + 512**2)
    logger.info(f"Formatted size: {size_str}")

    # Test timestamp
    timestamp = get_timestamp()
    logger.info(f"Timestamp: {timestamp}")

    # Test safe divide
    result = safe_divide(10, 2)
    logger.info(f"Safe divide (10/2): {result}")

    result = safe_divide(10, 0, default=999)
    logger.info(f"Safe divide (10/0): {result}")

    # Test clamp
    clamped = clamp(15, 0, 10)
    logger.info(f"Clamp (15, 0, 10): {clamped}")

    logger.info("Utility function tests passed!")


if __name__ == "__main__":
    setup_logging()
    test_utils()
