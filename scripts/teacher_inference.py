# SteveAI - Teacher Model Inference Script (GPU Optimized)
"""
This script runs teacher model inference to generate logits for knowledge distillation.
It should be run before training the student model.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the original teacher inference code
from teacher_inference_gpu import main

if __name__ == "__main__":
    main()
