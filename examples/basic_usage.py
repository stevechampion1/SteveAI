# SteveAI - Basic Usage Example

"""
This example demonstrates the basic usage of SteveAI for knowledge distillation.
It shows how to:
1. Run teacher inference to generate logits
2. Train a student model using the teacher logits
3. Evaluate the trained student model
4. Compare performance with the teacher model
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from steveai.core.data_utils import create_dataloader, prepare_student_dataset
from steveai.core.distillation_loss import DistillationLoss
from steveai.core.student_training import (
    StudentTrainingConfig,
    create_optimizer_and_scheduler,
    evaluate_model,
    load_student_model_and_tokenizer,
    train_epoch,
)
from steveai.evaluation.benchmark import PerformanceBenchmark
from steveai.evaluation.evaluate import ModelEvaluator

# Import SteveAI modules
from steveai.utils.config_manager import ConfigManager
from steveai.utils.utils import print_memory_usage, setup_logging


def main():
    """Main function demonstrating basic SteveAI usage."""

    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=== SteveAI Basic Usage Example ===")

    # Step 1: Load configuration
    logger.info("Step 1: Loading configuration...")
    config = ConfigManager("config.yaml")

    if not config.validate():
        logger.error("Configuration validation failed!")
        return

    # Get output paths
    paths = config.get_output_paths()
    logger.info(f"Output base directory: {paths['base_dir']}")

    # Step 2: Prepare dataset
    logger.info("Step 2: Preparing dataset...")

    # Check if teacher logits exist
    logits_dir = paths["logits_dir"]
    if not os.path.exists(logits_dir):
        logger.error(f"Teacher logits directory not found: {logits_dir}")
        logger.info("Please run teacher inference first using teacher_inference_gpu.py")
        return

    # Load student model and tokenizer
    logger.info("Loading student model and tokenizer...")
    model, tokenizer = load_student_model_and_tokenizer(StudentTrainingConfig())

    # Prepare dataset
    dataset = prepare_student_dataset(
        teacher_logits_path=logits_dir,
        tokenizer=tokenizer,
        max_seq_length=config.get("dataset.max_seq_length", 256),
        dataset_id=config.get("dataset.dataset_id", "yahma/alpaca-cleaned"),
        dataset_subset_size=config.get("dataset.dataset_subset_size", 500),
    )

    # Create data loaders
    from steveai.core.data_utils import create_data_splits

    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.get("training.batch_size", 8),
        shuffle=True,
        num_workers=config.get("inference.num_workers", 2),
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.get("training.batch_size", 8),
        shuffle=False,
        num_workers=config.get("inference.num_workers", 2),
    )

    logger.info(f"Dataset prepared: {len(dataset)} total samples")
    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Step 3: Setup training
    logger.info("Step 3: Setting up training...")

    # Create distillation loss function
    distillation_loss_fn = DistillationLoss()

    # Create optimizer and scheduler
    num_training_steps = len(train_dataloader) * config.get("training.num_epochs", 3)
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_training_steps)

    # Get device
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(f"Training on device: {device}")
    logger.info(f"Number of training steps: {num_training_steps}")

    # Step 4: Training loop
    logger.info("Step 4: Starting training...")

    num_epochs = config.get("training.num_epochs", 3)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            distillation_loss_fn=distillation_loss_fn,
            epoch=epoch,
            device=device,
        )

        # Evaluate model
        val_loss = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            distillation_loss_fn=distillation_loss_fn,
            device=device,
        )

        logger.info(
            f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = paths["checkpoint_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)

            from steveai.utils.model_utils import save_model_checkpoint

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                file_path=os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt"),
            )
            logger.info(f"Best model saved at epoch {epoch + 1}")

        print_memory_usage(f"After epoch {epoch + 1}")

    # Step 5: Save final model
    logger.info("Step 5: Saving final model...")

    final_model_dir = paths["final_model_dir"]
    os.makedirs(final_model_dir, exist_ok=True)

    from steveai.utils.model_utils import save_model_artifacts

    save_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        config=config.to_dict(),
        output_dir=final_model_dir,
        model_name="student_model",
    )

    logger.info(f"Final model saved to: {final_model_dir}")

    # Step 6: Evaluate final model
    logger.info("Step 6: Evaluating final model...")

    evaluator = ModelEvaluator(
        model_path=final_model_dir, tokenizer_path=final_model_dir, device="auto"
    )

    # Test prompts for generation evaluation
    test_prompts = [
        "Write a Python function to calculate the factorial of a number.",
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of using renewable energy?",
        "How does a neural network learn?",
        "Describe the process of photosynthesis.",
    ]

    # Comprehensive evaluation
    evaluation_results = evaluator.comprehensive_evaluation(
        dataloader=val_dataloader, test_prompts=test_prompts, max_samples=100
    )

    # Save evaluation results
    evaluation_dir = paths["evaluation_dir"]
    os.makedirs(evaluation_dir, exist_ok=True)

    import json

    with open(os.path.join(evaluation_dir, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)

    logger.info("Evaluation results saved")

    # Step 7: Performance benchmark
    logger.info("Step 7: Running performance benchmark...")

    benchmark = PerformanceBenchmark(
        model_path=final_model_dir, tokenizer_path=final_model_dir, device="auto"
    )

    benchmark_results = benchmark.comprehensive_benchmark(
        dataloader=val_dataloader, test_prompts=test_prompts, num_batches=10
    )

    # Save benchmark results
    benchmark_dir = paths["benchmark_dir"]
    os.makedirs(benchmark_dir, exist_ok=True)

    with open(os.path.join(benchmark_dir, "benchmark_results.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info("Benchmark results saved")

    # Step 8: Print summary
    logger.info("=== Training Summary ===")
    logger.info(f"Final validation loss: {best_val_loss:.4f}")

    if "perplexity" in evaluation_results:
        logger.info(
            f"Model perplexity: {evaluation_results['perplexity']['perplexity']:.4f}"
        )

    if "memory" in evaluation_results:
        logger.info(
            f"Model size: {evaluation_results['memory']['model_size_mb']:.2f} MB"
        )
        logger.info(
            f"Total parameters: {evaluation_results['memory']['total_parameters']:,}"
        )

    if "model_size" in benchmark_results:
        logger.info(
            f"Benchmark - Model size: {benchmark_results['model_size']['model_size_mb']:.2f} MB"
        )

    if "inference_speed" in benchmark_results:
        logger.info(
            f"Benchmark - Avg inference time: {benchmark_results['inference_speed']['avg_inference_time']:.4f}s"
        )
        logger.info(
            f"Benchmark - Tokens per second: {benchmark_results['inference_speed']['avg_tokens_per_second']:.2f}"
        )

    logger.info("=== Basic Usage Example Completed ===")


if __name__ == "__main__":
    main()
