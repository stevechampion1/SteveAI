# SteveAI - Configuration Manager

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager for SteveAI project."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    if self.config_path.endswith(".yaml") or self.config_path.endswith(
                        ".yml"
                    ):
                        self.config = yaml.safe_load(f)
                    elif self.config_path.endswith(".json"):
                        self.config = json.load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file format: {self.config_path}"
                        )

                logger.info(f"Loaded configuration from: {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                self.config = {}
        else:
            logger.info("No config file provided, using default configuration")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "teacher_model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "student_model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
                "tokenizer_path": "deepseek-ai/deepseek-coder-6.7b-instruct",
            },
            "dataset": {
                "dataset_id": "yahma/alpaca-cleaned",
                "dataset_subset_size": 500,
                "max_seq_length": 256,
                "train_split": "train",
            },
            "training": {
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "warmup_steps": 100,
                "temperature": 4.0,
                "alpha": 0.7,
                "beta": 0.3,
            },
            "inference": {"batch_size": 4, "num_workers": 2, "dtype": "float16"},
            "output": {
                "base_dir": "./output/SteveAI",
                "logits_dir": "teacher_logits_float16",
                "checkpoint_dir": "checkpoints",
                "final_model_dir": "final_model",
                "save_frequency": 20,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "log_memory_every": 20,
            },
            "environment": {
                "auto_detect": True,
                "kaggle_path": "/kaggle/working/SteveAI",
                "local_path": "./output/SteveAI",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'model.teacher_model_id')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'model.teacher_model_id')
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates
        """

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    isinstance(value, dict)
                    and key in base_dict
                    and isinstance(base_dict[key], dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.config, updates)

    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            path: Path to save configuration (uses self.config_path if None)
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path provided for saving configuration")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                if save_path.endswith(".yaml") or save_path.endswith(".yml"):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif save_path.endswith(".json"):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path}")

            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            raise

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate model configuration
            teacher_model_id = self.get("model.teacher_model_id")
            student_model_id = self.get("model.student_model_id")

            if not teacher_model_id or not student_model_id:
                logger.error("Teacher and student model IDs are required")
                return False

            # Validate training configuration
            learning_rate = self.get("training.learning_rate")
            if learning_rate is not None and learning_rate <= 0:
                logger.error("Learning rate must be positive")
                return False

            num_epochs = self.get("training.num_epochs")
            if num_epochs is not None and num_epochs <= 0:
                logger.error("Number of epochs must be positive")
                return False

            batch_size = self.get("training.batch_size")
            if batch_size is not None and batch_size <= 0:
                logger.error("Batch size must be positive")
                return False

            # Validate distillation configuration
            temperature = self.get("training.temperature")
            if temperature is not None and temperature <= 0:
                logger.error("Temperature must be positive")
                return False

            alpha = self.get("training.alpha")
            beta = self.get("training.beta")
            if alpha is not None and beta is not None:
                if not (0 <= alpha <= 1) or not (0 <= beta <= 1):
                    logger.error("Alpha and beta must be between 0 and 1")
                    return False

                if abs(alpha + beta - 1.0) > 1e-6:
                    logger.error("Alpha + beta must equal 1.0")
                    return False

            # Validate dataset configuration
            max_seq_length = self.get("dataset.max_seq_length")
            if max_seq_length is not None and max_seq_length <= 0:
                logger.error("Max sequence length must be positive")
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def get_output_paths(self) -> Dict[str, str]:
        """
        Get output paths based on environment.

        Returns:
            Dictionary of output paths
        """
        base_dir = self._get_output_base_dir()

        paths = {
            "base_dir": base_dir,
            "logits_dir": os.path.join(
                base_dir, self.get("output.logits_dir", "teacher_logits_float16")
            ),
            "checkpoint_dir": os.path.join(
                base_dir, self.get("output.checkpoint_dir", "checkpoints")
            ),
            "final_model_dir": os.path.join(
                base_dir, self.get("output.final_model_dir", "final_model")
            ),
            "evaluation_dir": os.path.join(base_dir, "evaluation_results"),
            "benchmark_dir": os.path.join(base_dir, "benchmark_results"),
        }

        return paths

    def _get_output_base_dir(self) -> str:
        """Get output base directory based on environment."""
        if self.get("environment.auto_detect", True):
            if os.path.exists("/kaggle/working"):
                return self.get("environment.kaggle_path", "/kaggle/working/SteveAI")
            else:
                return self.get("environment.local_path", "./output/SteveAI")
        else:
            return self.get("output.base_dir", "./output/SteveAI")

    def create_directories(self):
        """Create all necessary output directories."""
        paths = self.get_output_paths()

        for path_name, path in paths.items():
            if path_name != "base_dir":  # base_dir is parent of others
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            "student_model_id": self.get("model.student_model_id"),
            "tokenizer_path": self.get("model.tokenizer_path"),
            "learning_rate": self.get("training.learning_rate"),
            "num_epochs": self.get("training.num_epochs"),
            "batch_size": self.get("training.batch_size"),
            "gradient_accumulation_steps": self.get(
                "training.gradient_accumulation_steps"
            ),
            "max_grad_norm": self.get("training.max_grad_norm"),
            "warmup_steps": self.get("training.warmup_steps"),
            "temperature": self.get("training.temperature"),
            "alpha": self.get("training.alpha"),
            "beta": self.get("training.beta"),
            "max_seq_length": self.get("dataset.max_seq_length"),
            "num_workers": self.get("inference.num_workers"),
        }

    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference-specific configuration."""
        return {
            "teacher_model_id": self.get("model.teacher_model_id"),
            "tokenizer_path": self.get("model.tokenizer_path"),
            "dataset_id": self.get("dataset.dataset_id"),
            "dataset_subset_size": self.get("dataset.dataset_subset_size"),
            "max_seq_length": self.get("dataset.max_seq_length"),
            "batch_size": self.get("inference.batch_size"),
            "num_workers": self.get("inference.num_workers"),
            "dtype": self.get("inference.dtype"),
        }

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation-specific configuration."""
        return {
            "max_samples": self.get("evaluation.max_samples"),
            "test_prompts": self.get("evaluation.test_prompts", []),
            "generation_config": {
                "max_length": self.get("evaluation.generation.max_length", 256),
                "temperature": self.get("evaluation.generation.temperature", 0.7),
                "top_p": self.get("evaluation.generation.top_p", 0.9),
                "num_return_sequences": self.get(
                    "evaluation.generation.num_return_sequences", 1
                ),
            },
        }

    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmark-specific configuration."""
        return {
            "num_batches": self.get("benchmark.num_batches", 10),
            "warmup_batches": self.get("benchmark.warmup_batches", 2),
            "test_prompts": self.get("benchmark.test_prompts", []),
            "generation_config": {
                "max_length": self.get("benchmark.generation.max_length", 256),
                "temperature": self.get("benchmark.generation.temperature", 0.7),
                "top_p": self.get("benchmark.generation.top_p", 0.9),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)


def load_config_from_args(args: argparse.Namespace) -> ConfigManager:
    """
    Load configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        ConfigManager instance
    """
    config_manager = ConfigManager(args.config)

    # Override with command line arguments
    if hasattr(args, "learning_rate") and args.learning_rate is not None:
        config_manager.set("training.learning_rate", args.learning_rate)

    if hasattr(args, "batch_size") and args.batch_size is not None:
        config_manager.set("training.batch_size", args.batch_size)
        config_manager.set("inference.batch_size", args.batch_size)

    if hasattr(args, "num_epochs") and args.num_epochs is not None:
        config_manager.set("training.num_epochs", args.num_epochs)

    if hasattr(args, "max_seq_length") and args.max_seq_length is not None:
        config_manager.set("dataset.max_seq_length", args.max_seq_length)

    if hasattr(args, "temperature") and args.temperature is not None:
        config_manager.set("training.temperature", args.temperature)

    if hasattr(args, "alpha") and args.alpha is not None:
        config_manager.set("training.alpha", args.alpha)

    if hasattr(args, "beta") and args.beta is not None:
        config_manager.set("training.beta", args.beta)

    if hasattr(args, "output_dir") and args.output_dir is not None:
        config_manager.set("output.base_dir", args.output_dir)

    return config_manager


def create_config_template(output_path: str):
    """
    Create a configuration template file.

    Args:
        output_path: Path to save the template
    """
    config_manager = ConfigManager()
    config_manager.save(output_path)
    logger.info(f"Configuration template created at: {output_path}")


# Example usage and testing functions
def test_config_manager():
    """Test the configuration manager."""
    logger.info("Testing configuration manager...")

    # Test default configuration
    config = ConfigManager()
    logger.info(f"Default config loaded: {len(config.config)} sections")

    # Test get/set operations
    teacher_model = config.get("model.teacher_model_id")
    logger.info(f"Teacher model: {teacher_model}")

    config.set("model.teacher_model_id", "test-model")
    updated_model = config.get("model.teacher_model_id")
    logger.info(f"Updated teacher model: {updated_model}")

    # Test validation
    is_valid = config.validate()
    logger.info(f"Configuration valid: {is_valid}")

    # Test output paths
    paths = config.get_output_paths()
    logger.info(f"Output paths: {list(paths.keys())}")

    # Test specific configs
    training_config = config.get_training_config()
    logger.info(f"Training config keys: {list(training_config.keys())}")

    logger.info("Configuration manager tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_config_manager()
