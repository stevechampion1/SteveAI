# SteveAI - Model Evaluation Script

import argparse
import gc
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import default_data_collator

# Import our custom modules
from ..core.data_utils import create_dataloader, prepare_student_dataset
from ..utils.model_utils import load_model_checkpoint
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging


class ModelEvaluator:
    """Comprehensive model evaluation class."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        """
        Initialize the model evaluator.

        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer
            device: Device to use for evaluation
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

    def evaluate_perplexity(
        self, dataloader: DataLoader, max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model perplexity.

        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary containing perplexity metrics
        """
        self.logger.info("Evaluating perplexity...")

        total_loss = 0.0
        total_tokens = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Evaluating perplexity")
            ):
                if max_samples and num_samples >= max_samples:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                # Count non-ignored tokens
                non_ignored_tokens = (shift_labels != -100).sum().item()

                total_loss += loss.item()
                total_tokens += non_ignored_tokens
                num_samples += input_ids.size(0)

        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_samples": num_samples,
        }

        self.logger.info(f"Perplexity: {perplexity:.4f}")
        self.logger.info(f"Average loss: {avg_loss:.4f}")

        return metrics

    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to generate

        Returns:
            Dictionary containing generation metrics
        """
        self.logger.info("Evaluating generation quality...")

        generated_texts = []
        generation_times = []

        for prompt in tqdm(prompts, desc="Generating text"):
            start_time = time.time()

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            # Generate
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

            generation_time = time.time() - start_time
            generation_times.append(generation_time)

            # Decode generated text
            for output in outputs:
                generated_text = self.tokenizer.decode(
                    output[inputs["input_ids"].size(1) :], skip_special_tokens=True
                )
                generated_texts.append(generated_text)

        # Calculate metrics
        avg_generation_time = np.mean(generation_times)
        avg_text_length = np.mean([len(text.split()) for text in generated_texts])

        metrics = {
            "avg_generation_time": avg_generation_time,
            "avg_text_length": avg_text_length,
            "num_generated": len(generated_texts),
            "generated_texts": generated_texts[:5],  # Store first 5 examples
        }

        self.logger.info(f"Average generation time: {avg_generation_time:.4f}s")
        self.logger.info(f"Average text length: {avg_text_length:.2f} words")

        return metrics

    def evaluate_memory_usage(self) -> Dict[str, float]:
        """
        Evaluate model memory usage.

        Returns:
            Dictionary containing memory metrics
        """
        self.logger.info("Evaluating memory usage...")

        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        else:
            gpu_memory_allocated = 0.0
            gpu_memory_reserved = 0.0

        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1024**3

        metrics = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "gpu_memory_allocated_gb": gpu_memory_allocated,
            "gpu_memory_reserved_gb": gpu_memory_reserved,
            "cpu_memory_gb": cpu_memory,
        }

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"GPU memory allocated: {gpu_memory_allocated:.2f} GB")
        self.logger.info(f"CPU memory: {cpu_memory:.2f} GB")

        return metrics

    def compare_with_teacher(
        self,
        teacher_model_path: str,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compare student model with teacher model.

        Args:
            teacher_model_path: Path to teacher model
            dataloader: DataLoader for comparison
            max_samples: Maximum number of samples to compare

        Returns:
            Dictionary containing comparison metrics
        """
        self.logger.info("Comparing with teacher model...")

        # Load teacher model
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        teacher_model.eval()

        student_losses = []
        teacher_losses = []
        logit_similarities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Comparing models")
            ):
                if (
                    max_samples
                    and batch_idx * batch["input_ids"].size(0) >= max_samples
                ):
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Student model
                student_outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logits = student_outputs.logits

                # Teacher model
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

                # Calculate losses
                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                student_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                teacher_shift_logits = teacher_logits[..., :-1, :].contiguous()
                teacher_loss = F.cross_entropy(
                    teacher_shift_logits.view(-1, teacher_shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                student_losses.append(student_loss.item())
                teacher_losses.append(teacher_loss.item())

                # Calculate logit similarity (cosine similarity)
                student_logits_flat = student_logits.view(-1, student_logits.size(-1))
                teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

                similarity = F.cosine_similarity(
                    student_logits_flat, teacher_logits_flat, dim=-1
                ).mean()
                logit_similarities.append(similarity.item())

        # Clean up teacher model
        del teacher_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate metrics
        avg_student_loss = np.mean(student_losses)
        avg_teacher_loss = np.mean(teacher_losses)
        avg_logit_similarity = np.mean(logit_similarities)

        metrics = {
            "avg_student_loss": avg_student_loss,
            "avg_teacher_loss": avg_teacher_loss,
            "loss_ratio": (
                avg_student_loss / avg_teacher_loss
                if avg_teacher_loss > 0
                else float("inf")
            ),
            "avg_logit_similarity": avg_logit_similarity,
            "num_comparisons": len(student_losses),
        }

        self.logger.info(f"Average student loss: {avg_student_loss:.4f}")
        self.logger.info(f"Average teacher loss: {avg_teacher_loss:.4f}")
        self.logger.info(f"Loss ratio: {metrics['loss_ratio']:.4f}")
        self.logger.info(f"Average logit similarity: {avg_logit_similarity:.4f}")

        return metrics

    def comprehensive_evaluation(
        self,
        dataloader: DataLoader,
        teacher_model_path: Optional[str] = None,
        test_prompts: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.

        Args:
            dataloader: DataLoader for evaluation
            teacher_model_path: Path to teacher model for comparison
            test_prompts: List of prompts for generation testing
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary containing all evaluation metrics
        """
        self.logger.info("Starting comprehensive evaluation...")

        results = {}

        # Perplexity evaluation
        results["perplexity"] = self.evaluate_perplexity(dataloader, max_samples)

        # Memory usage evaluation
        results["memory"] = self.evaluate_memory_usage()

        # Generation quality evaluation
        if test_prompts:
            results["generation"] = self.evaluate_generation_quality(test_prompts)

        # Teacher comparison
        if teacher_model_path:
            results["teacher_comparison"] = self.compare_with_teacher(
                teacher_model_path, dataloader, max_samples
            )

        self.logger.info("Comprehensive evaluation completed")
        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SteveAI models")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default=None,
        help="Path to teacher model for comparison",
    )
    parser.add_argument(
        "--teacher_logits_path",
        type=str,
        required=True,
        help="Path to teacher logits directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use for evaluation"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Model Evaluation ---")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
        )

        # Prepare dataset
        logger.info("Preparing evaluation dataset...")
        dataset = prepare_student_dataset(
            teacher_logits_path=args.teacher_logits_path,
            tokenizer=evaluator.tokenizer,
            max_seq_length=256,
        )

        # Create dataloader
        dataloader = create_dataloader(
            dataset, batch_size=4, shuffle=False, num_workers=2
        )

        # Test prompts for generation evaluation
        test_prompts = [
            "Write a Python function to calculate the factorial of a number.",
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of using renewable energy?",
            "How does a neural network learn?",
            "Describe the process of photosynthesis.",
        ]

        # Perform comprehensive evaluation
        results = evaluator.comprehensive_evaluation(
            dataloader=dataloader,
            teacher_model_path=args.teacher_model_path,
            test_prompts=test_prompts,
            max_samples=args.max_samples,
        )

        # Save results
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to: {results_path}")

        # Print summary
        logger.info("--- Evaluation Summary ---")
        if "perplexity" in results:
            logger.info(f"Perplexity: {results['perplexity']['perplexity']:.4f}")

        if "memory" in results:
            logger.info(f"Total parameters: {results['memory']['total_parameters']:,}")
            logger.info(
                f"GPU memory: {results['memory']['gpu_memory_allocated_gb']:.2f} GB"
            )

        if "teacher_comparison" in results:
            logger.info(
                f"Loss ratio (student/teacher): {results['teacher_comparison']['loss_ratio']:.4f}"
            )
            logger.info(
                f"Logit similarity: {results['teacher_comparison']['avg_logit_similarity']:.4f}"
            )

        logger.info("--- Evaluation Completed ---")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if "evaluator" in locals():
            del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage("Final")


if __name__ == "__main__":
    main()
