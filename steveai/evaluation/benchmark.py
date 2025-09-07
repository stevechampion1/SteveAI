# SteveAI - Performance Benchmark Script

import argparse
import gc
import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our custom modules
from ..core.data_utils import create_dataloader, prepare_student_dataset
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging


class PerformanceBenchmark:
    """Comprehensive performance benchmarking class."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        """
        Initialize the performance benchmark.

        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer
            device: Device to use for benchmarking
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        self.logger.info(f"Loading model from: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        self.logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        end_time = time.time()
        return end_time - start_time

    def benchmark_inference_speed(
        self, dataloader: DataLoader, num_batches: int = 10, warmup_batches: int = 2
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            dataloader: DataLoader for benchmarking
            num_batches: Number of batches to benchmark
            warmup_batches: Number of warmup batches

        Returns:
            Dictionary containing speed metrics
        """
        self.logger.info("Benchmarking inference speed...")

        times = []
        tokens_per_second = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Benchmarking inference")
            ):
                if batch_idx >= num_batches + warmup_batches:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Time the forward pass
                start_time = time.time()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                end_time = time.time()

                # Skip warmup batches
                if batch_idx >= warmup_batches:
                    batch_time = end_time - start_time
                    times.append(batch_time)

                    # Calculate tokens per second
                    num_tokens = input_ids.numel()
                    tps = num_tokens / batch_time
                    tokens_per_second.append(tps)

        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        avg_tps = statistics.mean(tokens_per_second)
        std_tps = (
            statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0.0
        )

        metrics = {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "avg_tokens_per_second": avg_tps,
            "std_tokens_per_second": std_tps,
            "min_inference_time": min(times),
            "max_inference_time": max(times),
            "num_batches": len(times),
        }

        self.logger.info(f"Average inference time: {avg_time:.4f}s ± {std_time:.4f}s")
        self.logger.info(f"Average tokens per second: {avg_tps:.2f} ± {std_tps:.2f}")

        return metrics

    def benchmark_generation_speed(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> Dict[str, float]:
        """
        Benchmark text generation speed.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to generate

        Returns:
            Dictionary containing generation speed metrics
        """
        self.logger.info("Benchmarking generation speed...")

        generation_times = []
        tokens_generated = []

        for prompt in tqdm(prompts, desc="Benchmarking generation"):
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            input_length = inputs["input_ids"].size(1)

            # Time generation
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            end_time = time.time()

            generation_time = end_time - start_time
            generation_times.append(generation_time)

            # Count generated tokens
            for output in outputs:
                generated_tokens = output[input_length:].numel()
                tokens_generated.append(generated_tokens)

        # Calculate statistics
        avg_generation_time = statistics.mean(generation_times)
        std_generation_time = (
            statistics.stdev(generation_times) if len(generation_times) > 1 else 0.0
        )
        avg_tokens_generated = statistics.mean(tokens_generated)
        avg_generation_tps = (
            avg_tokens_generated / avg_generation_time
            if avg_generation_time > 0
            else 0.0
        )

        metrics = {
            "avg_generation_time": avg_generation_time,
            "std_generation_time": std_generation_time,
            "avg_tokens_generated": avg_tokens_generated,
            "avg_generation_tps": avg_generation_tps,
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times),
            "num_prompts": len(prompts),
        }

        self.logger.info(
            f"Average generation time: {avg_generation_time:.4f}s ± {std_generation_time:.4f}s"
        )
        self.logger.info(f"Average generation TPS: {avg_generation_tps:.2f}")

        return metrics

    def benchmark_memory_usage(
        self, dataloader: DataLoader, num_batches: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark memory usage during inference.

        Args:
            dataloader: DataLoader for benchmarking
            num_batches: Number of batches to benchmark

        Returns:
            Dictionary containing memory metrics
        """
        self.logger.info("Benchmarking memory usage...")

        memory_usage = []

        # Initial memory
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**3
        else:
            initial_gpu_memory = 0.0

        process = psutil.Process(os.getpid())
        initial_cpu_memory = process.memory_info().rss / 1024**3

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Benchmarking memory")
            ):
                if batch_idx >= num_batches:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Measure memory usage
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                else:
                    gpu_memory = 0.0

                cpu_memory = process.memory_info().rss / 1024**3

                memory_usage.append(
                    {"gpu_memory": gpu_memory, "cpu_memory": cpu_memory}
                )

                # Cleanup
                del outputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate statistics
        gpu_memories = [m["gpu_memory"] for m in memory_usage]
        cpu_memories = [m["cpu_memory"] for m in memory_usage]

        metrics = {
            "initial_gpu_memory_gb": initial_gpu_memory,
            "initial_cpu_memory_gb": initial_cpu_memory,
            "avg_gpu_memory_gb": statistics.mean(gpu_memories),
            "max_gpu_memory_gb": max(gpu_memories),
            "avg_cpu_memory_gb": statistics.mean(cpu_memories),
            "max_cpu_memory_gb": max(cpu_memories),
            "gpu_memory_peak_gb": (
                torch.cuda.max_memory_allocated(self.device) / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
        }

        self.logger.info(f"Average GPU memory: {metrics['avg_gpu_memory_gb']:.2f} GB")
        self.logger.info(f"Peak GPU memory: {metrics['gpu_memory_peak_gb']:.2f} GB")
        self.logger.info(f"Average CPU memory: {metrics['avg_cpu_memory_gb']:.2f} GB")

        return metrics

    def benchmark_model_size(self) -> Dict[str, Any]:
        """
        Benchmark model size and parameters.

        Returns:
            Dictionary containing model size metrics
        """
        self.logger.info("Benchmarking model size...")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Estimate model size in MB
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / 1024**2

        # Get model configuration
        config = self.model.config
        model_info = {
            "model_type": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", 0),
            "num_layers": getattr(config, "num_hidden_layers", 0),
            "num_attention_heads": getattr(config, "num_attention_heads", 0),
            "vocab_size": getattr(config, "vocab_size", 0),
        }

        metrics = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "model_size_gb": model_size_mb / 1024,
            "model_info": model_info,
        }

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(
            f"Model size: {model_size_mb:.2f} MB ({model_size_mb/1024:.2f} GB)"
        )

        return metrics

    def compare_models(
        self,
        other_model_path: str,
        other_tokenizer_path: str,
        dataloader: DataLoader,
        num_batches: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare performance with another model.

        Args:
            other_model_path: Path to the other model
            other_tokenizer_path: Path to the other model's tokenizer
            dataloader: DataLoader for comparison
            num_batches: Number of batches to compare

        Returns:
            Dictionary containing comparison metrics
        """
        self.logger.info("Comparing with another model...")

        # Load other model
        other_model = AutoModelForCausalLM.from_pretrained(
            other_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        other_model.eval()

        # Benchmark current model
        current_metrics = self.benchmark_inference_speed(dataloader, num_batches)
        current_memory = self.benchmark_memory_usage(dataloader, num_batches)
        current_size = self.benchmark_model_size()

        # Benchmark other model
        other_times = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Benchmarking other model")
            ):
                if batch_idx >= num_batches:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                start_time = time.time()
                outputs = other_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                end_time = time.time()

                other_times.append(end_time - start_time)

        # Calculate other model metrics
        other_avg_time = statistics.mean(other_times)
        other_total_params = sum(p.numel() for p in other_model.parameters())

        # Clean up other model
        del other_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate comparison metrics
        speed_ratio = other_avg_time / current_metrics["avg_inference_time"]
        size_ratio = other_total_params / current_size["total_parameters"]

        comparison = {
            "current_model": {
                "avg_inference_time": current_metrics["avg_inference_time"],
                "total_parameters": current_size["total_parameters"],
                "model_size_mb": current_size["model_size_mb"],
            },
            "other_model": {
                "avg_inference_time": other_avg_time,
                "total_parameters": other_total_params,
            },
            "speed_ratio": speed_ratio,
            "size_ratio": size_ratio,
            "speedup": speed_ratio if speed_ratio > 1 else 1 / speed_ratio,
            "compression_ratio": size_ratio if size_ratio > 1 else 1 / size_ratio,
        }

        self.logger.info(f"Speed ratio (other/current): {speed_ratio:.2f}")
        self.logger.info(f"Size ratio (other/current): {size_ratio:.2f}")

        return comparison

    def comprehensive_benchmark(
        self,
        dataloader: DataLoader,
        test_prompts: Optional[List[str]] = None,
        other_model_path: Optional[str] = None,
        other_tokenizer_path: Optional[str] = None,
        num_batches: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive benchmarking.

        Args:
            dataloader: DataLoader for benchmarking
            test_prompts: List of prompts for generation testing
            other_model_path: Path to model for comparison
            other_tokenizer_path: Path to other model's tokenizer
            num_batches: Number of batches to benchmark

        Returns:
            Dictionary containing all benchmark results
        """
        self.logger.info("Starting comprehensive benchmark...")

        results = {}

        # Model size benchmark
        results["model_size"] = self.benchmark_model_size()

        # Inference speed benchmark
        results["inference_speed"] = self.benchmark_inference_speed(
            dataloader, num_batches
        )

        # Memory usage benchmark
        results["memory_usage"] = self.benchmark_memory_usage(dataloader, num_batches)

        # Generation speed benchmark
        if test_prompts:
            results["generation_speed"] = self.benchmark_generation_speed(test_prompts)

        # Model comparison
        if other_model_path and other_tokenizer_path:
            results["model_comparison"] = self.compare_models(
                other_model_path, other_tokenizer_path, dataloader, num_batches
            )

        self.logger.info("Comprehensive benchmark completed")
        return results


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark SteveAI models")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to benchmark"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--teacher_logits_path",
        type=str,
        required=True,
        help="Path to teacher logits directory",
    )
    parser.add_argument(
        "--other_model_path",
        type=str,
        default=None,
        help="Path to another model for comparison",
    )
    parser.add_argument(
        "--other_tokenizer_path",
        type=str,
        default=None,
        help="Path to other model's tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_batches", type=int, default=10, help="Number of batches to benchmark"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use for benchmarking"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Model Benchmark ---")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize benchmark
        benchmark = PerformanceBenchmark(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
        )

        # Prepare dataset
        logger.info("Preparing benchmark dataset...")
        dataset = prepare_student_dataset(
            teacher_logits_path=args.teacher_logits_path,
            tokenizer=benchmark.tokenizer,
            max_seq_length=256,
        )

        # Create dataloader
        dataloader = create_dataloader(
            dataset, batch_size=4, shuffle=False, num_workers=2
        )

        # Test prompts for generation benchmarking
        test_prompts = [
            "Write a Python function to calculate the factorial of a number.",
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of using renewable energy?",
            "How does a neural network learn?",
            "Describe the process of photosynthesis.",
        ]

        # Perform comprehensive benchmark
        results = benchmark.comprehensive_benchmark(
            dataloader=dataloader,
            test_prompts=test_prompts,
            other_model_path=args.other_model_path,
            other_tokenizer_path=args.other_tokenizer_path,
            num_batches=args.num_batches,
        )

        # Save results
        results_path = os.path.join(args.output_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results saved to: {results_path}")

        # Print summary
        logger.info("--- Benchmark Summary ---")
        if "model_size" in results:
            logger.info(f"Model size: {results['model_size']['model_size_mb']:.2f} MB")
            logger.info(
                f"Total parameters: {results['model_size']['total_parameters']:,}"
            )

        if "inference_speed" in results:
            logger.info(
                f"Average inference time: {results['inference_speed']['avg_inference_time']:.4f}s"
            )
            logger.info(
                f"Average tokens per second: {results['inference_speed']['avg_tokens_per_second']:.2f}"
            )

        if "memory_usage" in results:
            logger.info(
                f"Peak GPU memory: {results['memory_usage']['gpu_memory_peak_gb']:.2f} GB"
            )

        if "model_comparison" in results:
            logger.info(f"Speedup: {results['model_comparison']['speedup']:.2f}x")
            logger.info(
                f"Compression ratio: {results['model_comparison']['compression_ratio']:.2f}x"
            )

        logger.info("--- Benchmark Completed ---")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if "benchmark" in locals():
            del benchmark
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage("Final")


if __name__ == "__main__":
    main()
