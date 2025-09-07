# SteveAI - Knowledge Distillation Framework
"""
SteveAI is a comprehensive knowledge distillation framework for training
smaller student models from larger teacher models.

This package provides:
- Core distillation algorithms and loss functions
- Data processing and dataset utilities
- Model training and evaluation tools
- Performance benchmarking and optimization
- Deployment and monitoring capabilities
"""

__version__ = "1.0.0"
__author__ = "SteveAI Team"
__email__ = "steveai@example.com"

from .core.data_utils import (
    StudentDataset,
    TeacherLogitsLoader,
    prepare_student_dataset,
)

# Import core modules
from .core.distillation_loss import (
    AdvancedDistillationLoss,
    DistillationLoss,
    FocalDistillationLoss,
)
from .utils.config_manager import ConfigManager
from .utils.model_utils import (
    get_model_summary,
    load_model_checkpoint,
    save_model_checkpoint,
)

__all__ = [
    "DistillationLoss",
    "AdvancedDistillationLoss",
    "FocalDistillationLoss",
    "TeacherLogitsLoader",
    "StudentDataset",
    "prepare_student_dataset",
    "ConfigManager",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "get_model_summary",
]
