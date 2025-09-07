# SteveAI - Model Optimization Script

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.onnx
import torch.quantization as quantization

from ..utils.config_manager import ConfigManager

# Import our custom modules
from ..utils.model_utils import (
    get_model_summary,
    load_model_artifacts,
    save_model_artifacts,
)
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Model optimization utilities for SteveAI."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize the model optimizer.

        Args:
            model: Model to optimize
            device: Device to use for optimization
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.original_model = None

        logger.info(f"Model optimizer initialized on device: {self.device}")

    def quantize_model(
        self,
        quantization_type: str = "dynamic",
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Quantize the model to reduce size and improve inference speed.

        Args:
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
            calibration_data: Data for static quantization calibration

        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model with {quantization_type} quantization...")

        # Save original model
        self.original_model = self.model

        if quantization_type == "dynamic":
            return self._dynamic_quantization()
        elif quantization_type == "static":
            return self._static_quantization(calibration_data)
        elif quantization_type == "qat":
            return self._qat_quantization()
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")

    def _dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )

        logger.info("Dynamic quantization applied")
        return quantized_model

    def _static_quantization(
        self, calibration_data: Optional[List[torch.Tensor]]
    ) -> nn.Module:
        """Apply static quantization."""
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")

        # Set quantization config
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        self.model.qconfig = quantization_config

        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(self.model)

        # Calibrate with data
        prepared_model.eval()
        with torch.no_grad():
            for data in calibration_data:
                if isinstance(data, (list, tuple)):
                    prepared_model(*data)
                else:
                    prepared_model(data)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)

        logger.info("Static quantization applied")
        return quantized_model

    def _qat_quantization(self) -> nn.Module:
        """Apply quantization-aware training."""
        # Set quantization config
        quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
        self.model.qconfig = quantization_config

        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(self.model)

        logger.info("Model prepared for quantization-aware training")
        return prepared_model

    def prune_model(
        self, pruning_ratio: float = 0.2, pruning_type: str = "magnitude"
    ) -> nn.Module:
        """
        Prune the model to reduce parameters.

        Args:
            pruning_ratio: Ratio of parameters to prune
            pruning_type: Type of pruning ('magnitude', 'random')

        Returns:
            Pruned model
        """
        logger.info(
            f"Pruning model with {pruning_type} pruning (ratio: {pruning_ratio})..."
        )

        # Save original model
        self.original_model = self.model

        if pruning_type == "magnitude":
            return self._magnitude_pruning(pruning_ratio)
        elif pruning_type == "random":
            return self._random_pruning(pruning_ratio)
        else:
            raise ValueError(f"Unsupported pruning type: {pruning_type}")

    def _magnitude_pruning(self, pruning_ratio: float) -> nn.Module:
        """Apply magnitude-based pruning."""
        import torch.nn.utils.prune as prune

        # Prune linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_ratio)

        # Make pruning permanent
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")

        logger.info("Magnitude-based pruning applied")
        return self.model

    def _random_pruning(self, pruning_ratio: float) -> nn.Module:
        """Apply random pruning."""
        import torch.nn.utils.prune as prune

        # Prune linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.random_unstructured(module, name="weight", amount=pruning_ratio)

        # Make pruning permanent
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, "weight")

        logger.info("Random pruning applied")
        return self.model

    def optimize_for_inference(self) -> nn.Module:
        """
        Optimize model for inference.

        Returns:
            Optimized model
        """
        logger.info("Optimizing model for inference...")

        # Set to evaluation mode
        self.model.eval()

        # Apply optimizations
        if hasattr(torch.jit, "optimize_for_inference"):
            self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model))
        else:
            # Fallback: just script the model
            self.model = torch.jit.script(self.model)

        logger.info("Model optimized for inference")
        return self.model

    def export_to_onnx(
        self, input_shape: Tuple[int, ...], output_path: str, opset_version: int = 11
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        logger.info(f"Exporting model to ONNX: {output_path}")

        self.model.eval()
        self.model = self.model.to(self.device)

        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)

        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        logger.info(f"Model exported to ONNX: {output_path}")

    def export_to_torchscript(self, output_path: str) -> None:
        """
        Export model to TorchScript format.

        Args:
            output_path: Path to save TorchScript model
        """
        logger.info(f"Exporting model to TorchScript: {output_path}")

        self.model.eval()

        # Convert to TorchScript
        scripted_model = torch.jit.script(self.model)

        # Save
        scripted_model.save(output_path)

        logger.info(f"Model exported to TorchScript: {output_path}")

    def compare_models(self, optimized_model: nn.Module) -> Dict[str, Any]:
        """
        Compare original and optimized models.

        Args:
            optimized_model: Optimized model to compare

        Returns:
            Comparison results
        """
        if self.original_model is None:
            logger.warning("No original model available for comparison")
            return {}

        original_summary = get_model_summary(self.original_model)
        optimized_summary = get_model_summary(optimized_model)

        comparison = {
            "original": original_summary,
            "optimized": optimized_summary,
            "size_reduction_mb": original_summary["model_size_mb"]
            - optimized_summary["model_size_mb"],
            "size_reduction_percent": (
                1
                - optimized_summary["model_size_mb"] / original_summary["model_size_mb"]
            )
            * 100,
            "parameter_reduction": original_summary["total_parameters"]
            - optimized_summary["total_parameters"],
            "parameter_reduction_percent": (
                1
                - optimized_summary["total_parameters"]
                / original_summary["total_parameters"]
            )
            * 100,
        }

        logger.info(
            f"Size reduction: {comparison['size_reduction_mb']:.2f} MB ({comparison['size_reduction_percent']:.2f}%)"
        )
        logger.info(
            f"Parameter reduction: {comparison['parameter_reduction']:,} ({comparison['parameter_reduction_percent']:.2f}%)"
        )

        return comparison


def benchmark_model_performance(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model performance.

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Performance metrics
    """
    logger.info("Benchmarking model performance...")

    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    metrics = {
        "avg_inference_time": avg_time,
        "min_inference_time": min_time,
        "max_inference_time": max_time,
        "throughput_fps": 1.0 / avg_time,
        "num_iterations": num_iterations,
    }

    logger.info(f"Average inference time: {avg_time:.4f}s")
    logger.info(f"Throughput: {metrics['throughput_fps']:.2f} FPS")

    return metrics


def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize SteveAI models")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to optimize"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./optimized_models",
        help="Output directory for optimized models",
    )
    parser.add_argument(
        "--optimization_type",
        type=str,
        default="quantization",
        choices=["quantization", "pruning", "inference", "all"],
        help="Type of optimization to apply",
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "qat"],
        help="Type of quantization",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.2, help="Pruning ratio"
    )
    parser.add_argument(
        "--pruning_type",
        type=str,
        default="magnitude",
        choices=["magnitude", "random"],
        help="Type of pruning",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[256],
        help="Input shape for optimization",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark model performance"
    )
    parser.add_argument(
        "--export_formats",
        type=str,
        nargs="+",
        choices=["onnx", "torchscript"],
        help="Export formats",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Model Optimization ---")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Initialize optimizer
        optimizer = ModelOptimizer(model)

        # Apply optimizations
        if args.optimization_type in ["quantization", "all"]:
            logger.info("Applying quantization...")
            optimized_model = optimizer.quantize_model(args.quantization_type)

            # Save quantized model
            quantized_path = os.path.join(args.output_dir, "quantized_model")
            os.makedirs(quantized_path, exist_ok=True)
            optimized_model.save_pretrained(quantized_path)

            # Compare models
            comparison = optimizer.compare_models(optimized_model)
            comparison_path = os.path.join(
                args.output_dir, "quantization_comparison.json"
            )
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)

        if args.optimization_type in ["pruning", "all"]:
            logger.info("Applying pruning...")
            optimized_model = optimizer.prune_model(
                args.pruning_ratio, args.pruning_type
            )

            # Save pruned model
            pruned_path = os.path.join(args.output_dir, "pruned_model")
            os.makedirs(pruned_path, exist_ok=True)
            optimized_model.save_pretrained(pruned_path)

            # Compare models
            comparison = optimizer.compare_models(optimized_model)
            comparison_path = os.path.join(args.output_dir, "pruning_comparison.json")
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)

        if args.optimization_type in ["inference", "all"]:
            logger.info("Optimizing for inference...")
            optimized_model = optimizer.optimize_for_inference()

            # Save optimized model
            inference_path = os.path.join(args.output_dir, "inference_optimized")
            os.makedirs(inference_path, exist_ok=True)
            torch.save(optimized_model, os.path.join(inference_path, "model.pt"))

        # Export to different formats
        if args.export_formats:
            for export_format in args.export_formats:
                if export_format == "onnx":
                    onnx_path = os.path.join(args.output_dir, "model.onnx")
                    optimizer.export_to_onnx(tuple(args.input_shape), onnx_path)

                elif export_format == "torchscript":
                    torchscript_path = os.path.join(args.output_dir, "model.pt")
                    optimizer.export_to_torchscript(torchscript_path)

        # Benchmark performance
        if args.benchmark:
            logger.info("Benchmarking model performance...")
            performance = benchmark_model_performance(model, tuple(args.input_shape))

            performance_path = os.path.join(
                args.output_dir, "performance_benchmark.json"
            )
            with open(performance_path, "w") as f:
                json.dump(performance, f, indent=2)

        logger.info("--- Model Optimization Completed ---")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
