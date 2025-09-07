# SteveAI - Model Utility Functions

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    file_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        loss: Current loss value
        file_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": time.time(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if additional_info is not None:
        checkpoint["additional_info"] = additional_info

    torch.save(checkpoint, file_path)
    logger.info(f"Model checkpoint saved to: {file_path}")


def load_model_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    file_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        file_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Dictionary containing checkpoint information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(file_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Model checkpoint loaded from: {file_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return checkpoint


def save_model_weights(model: nn.Module, file_path: str) -> None:
    """
    Save only model weights.

    Args:
        model: Model to save
        file_path: Path to save weights
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    logger.info(f"Model weights saved to: {file_path}")


def load_model_weights(
    model: nn.Module, file_path: str, device: Optional[torch.device] = None
) -> None:
    """
    Load only model weights.

    Args:
        model: Model to load weights into
        file_path: Path to weights file
        device: Device to load weights on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Model weights loaded from: {file_path}")


def save_model_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save model configuration.

    Args:
        config: Configuration dictionary
        file_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model configuration saved to: {file_path}")


def load_model_config(file_path: str) -> Dict[str, Any]:
    """
    Load model configuration.

    Args:
        file_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(file_path, "r") as f:
        config = json.load(f)

    logger.info(f"Model configuration loaded from: {file_path}")
    return config


def freeze_model_parameters(
    model: nn.Module, freeze_layers: Optional[List[str]] = None
) -> None:
    """
    Freeze model parameters.

    Args:
        model: Model to freeze
        freeze_layers: List of layer names to freeze (None to freeze all)
    """
    if freeze_layers is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("All model parameters frozen")
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
        logger.info(f"Frozen layers: {freeze_layers}")


def unfreeze_model_parameters(
    model: nn.Module, unfreeze_layers: Optional[List[str]] = None
) -> None:
    """
    Unfreeze model parameters.

    Args:
        model: Model to unfreeze
        unfreeze_layers: List of layer names to unfreeze (None to unfreeze all)
    """
    if unfreeze_layers is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        logger.info("All model parameters unfrozen")
    else:
        # Unfreeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
        logger.info(f"Unfrozen layers: {unfreeze_layers}")


def get_trainable_parameters(model: nn.Module) -> List[torch.Tensor]:
    """
    Get list of trainable parameters.

    Args:
        model: Model to get parameters from

    Returns:
        List of trainable parameters
    """
    return [param for param in model.parameters() if param.requires_grad]


def get_frozen_parameters(model: nn.Module) -> List[torch.Tensor]:
    """
    Get list of frozen parameters.

    Args:
        model: Model to get parameters from

    Returns:
        List of frozen parameters
    """
    return [param for param in model.parameters() if not param.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.

    Args:
        model: Model to count parameters from

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: nn.Module) -> int:
    """
    Count frozen parameters.

    Args:
        model: Model to count parameters from

    Returns:
        Number of frozen parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get model summary information.

    Args:
        model: Model to summarize

    Returns:
        Dictionary containing model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_parameters(model)
    frozen_params = count_frozen_parameters(model)

    # Get model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2

    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "model_size_mb": model_size_mb,
        "model_size_gb": model_size_mb / 1024,
        "trainable_percentage": (
            (trainable_params / total_params * 100) if total_params > 0 else 0
        ),
    }

    return summary


def print_model_summary(model: nn.Module) -> None:
    """
    Print model summary.

    Args:
        model: Model to summarize
    """
    summary = get_model_summary(model)

    logger.info("=== Model Summary ===")
    logger.info(f"Total parameters: {summary['total_parameters']:,}")
    logger.info(f"Trainable parameters: {summary['trainable_parameters']:,}")
    logger.info(f"Frozen parameters: {summary['frozen_parameters']:,}")
    logger.info(f"Trainable percentage: {summary['trainable_percentage']:.2f}%")
    logger.info(
        f"Model size: {summary['model_size_mb']:.2f} MB ({summary['model_size_gb']:.2f} GB)"
    )


def copy_model_weights(source_model: nn.Module, target_model: nn.Module) -> None:
    """
    Copy weights from source model to target model.

    Args:
        source_model: Source model
        target_model: Target model
    """
    target_model.load_state_dict(source_model.state_dict())
    logger.info("Model weights copied from source to target")


def initialize_weights(model: nn.Module, init_type: str = "xavier_uniform") -> None:
    """
    Initialize model weights.

    Args:
        model: Model to initialize
        init_type: Initialization type ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """

    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.1)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(init_weights)
    logger.info(f"Model weights initialized with {init_type}")


def get_gradient_norm(model: nn.Module) -> float:
    """
    Get gradient norm of model parameters.

    Args:
        model: Model to get gradient norm from

    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradients of model parameters.

    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm

    Returns:
        Actual gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def save_model_artifacts(
    model: nn.Module,
    tokenizer: Any,
    config: Dict[str, Any],
    output_dir: str,
    model_name: str = "model",
) -> None:
    """
    Save complete model artifacts.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        config: Model configuration
        output_dir: Output directory
        model_name: Name for the model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f"{model_name}_pytorch_model.bin")
    save_model_weights(model, model_path)

    # Save tokenizer
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    save_model_config(config, config_path)

    # Save model summary
    summary = get_model_summary(model)
    summary_path = os.path.join(output_dir, "model_summary.json")
    save_model_config(summary, summary_path)

    logger.info(f"Model artifacts saved to: {output_dir}")


def load_model_artifacts(
    model: nn.Module,
    tokenizer: Any,
    model_dir: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load complete model artifacts.

    Args:
        model: Model to load weights into
        tokenizer: Tokenizer to load
        model_dir: Model directory
        device: Device to load on

    Returns:
        Model configuration
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model weights
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model_pytorch_model.bin")

    if os.path.exists(model_path):
        load_model_weights(model, model_path, device)

    # Load tokenizer
    if hasattr(tokenizer, "from_pretrained"):
        tokenizer = tokenizer.from_pretrained(model_dir)

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        config = load_model_config(config_path)

    logger.info(f"Model artifacts loaded from: {model_dir}")
    return config


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """
    Compare two models.

    Args:
        model1: First model
        model2: Second model

    Returns:
        Comparison results
    """
    summary1 = get_model_summary(model1)
    summary2 = get_model_summary(model2)

    comparison = {
        "model1": summary1,
        "model2": summary2,
        "parameter_difference": summary1["total_parameters"]
        - summary2["total_parameters"],
        "size_difference_mb": summary1["model_size_mb"] - summary2["model_size_mb"],
        "compression_ratio": (
            summary1["total_parameters"] / summary2["total_parameters"]
            if summary2["total_parameters"] > 0
            else float("inf")
        ),
    }

    return comparison


def export_model_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    device: Optional[torch.device] = None,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: Model to export
        input_shape: Input tensor shape
        output_path: Path to save ONNX model
        device: Device to export on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(f"Model exported to ONNX: {output_path}")


# Example usage and testing functions
def test_model_utils():
    """Test model utility functions."""
    logger.info("Testing model utility functions...")

    # Create a simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    model = TestModel()

    # Test model summary
    summary = get_model_summary(model)
    logger.info(f"Model summary: {summary}")

    # Test parameter counting
    trainable = count_trainable_parameters(model)
    frozen = count_frozen_parameters(model)
    logger.info(f"Trainable parameters: {trainable}, Frozen parameters: {frozen}")

    # Test freezing/unfreezing
    freeze_model_parameters(model, ["linear1"])
    trainable_after_freeze = count_trainable_parameters(model)
    logger.info(
        f"Trainable parameters after freezing linear1: {trainable_after_freeze}"
    )

    unfreeze_model_parameters(model, ["linear1"])
    trainable_after_unfreeze = count_trainable_parameters(model)
    logger.info(
        f"Trainable parameters after unfreezing linear1: {trainable_after_unfreeze}"
    )

    # Test weight initialization
    initialize_weights(model, "xavier_uniform")
    logger.info("Model weights initialized")

    logger.info("Model utility function tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model_utils()
