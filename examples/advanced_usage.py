# SteveAI - Advanced Usage Example

"""
This example demonstrates advanced usage of SteveAI including:
1. Custom distillation loss functions
2. Model optimization (quantization, pruning)
3. Advanced monitoring and visualization
4. Model deployment
5. Performance comparison
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark import PerformanceBenchmark

# Import SteveAI modules
from config_manager import ConfigManager
from data_utils import create_data_splits, create_dataloader, prepare_student_dataset
from deploy import FastAPIModelServer, ModelServer
from distillation_loss import (
    AdvancedDistillationLoss,
    DistillationLoss,
    FocalDistillationLoss,
)
from evaluate import ModelEvaluator
from monitor_training import RealTimeMonitor, TrainingMonitor
from optimize_model import ModelOptimizer
from student_training import (
    StudentTrainingConfig,
    create_optimizer_and_scheduler,
    evaluate_model,
    load_student_model_and_tokenizer,
    train_epoch,
)
from utils import print_memory_usage, setup_logging
from visualize_results import ResultsVisualizer


class CustomDistillationLoss(nn.Module):
    """Custom distillation loss with additional regularization."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.base_loss = DistillationLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, student_logits, teacher_logits, labels, **kwargs):
        # Base distillation loss
        dist_loss, hard_loss = self.base_loss(
            student_logits, teacher_logits, labels, **kwargs
        )

        # Add L2 regularization on logits
        l2_reg = self.mse_loss(student_logits, torch.zeros_like(student_logits)).mean()

        # Combine losses
        total_dist_loss = dist_loss + 0.01 * l2_reg

        if self.reduction == "mean":
            return total_dist_loss.mean(), hard_loss.mean()
        elif self.reduction == "sum":
            return total_dist_loss.sum(), hard_loss.sum()
        else:
            return total_dist_loss, hard_loss


def advanced_training_example():
    """Demonstrate advanced training with custom loss and monitoring."""

    logger = logging.getLogger(__name__)
    logger.info("=== Advanced Training Example ===")

    # Load configuration
    config = ConfigManager("config.yaml")
    paths = config.get_output_paths()

    # Setup monitoring
    monitor = TrainingMonitor(
        log_dir=os.path.join(paths["base_dir"], "training_logs"),
        update_interval=10,
        save_plots=True,
    )
    monitor.start_monitoring()

    # Load model and data
    model, tokenizer = load_student_model_and_tokenizer(StudentTrainingConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare dataset
    dataset = prepare_student_dataset(
        teacher_logits_path=paths["logits_dir"],
        tokenizer=tokenizer,
        max_seq_length=config.get("dataset.max_seq_length", 256),
    )

    train_dataset, val_dataset, _ = create_data_splits(
        dataset, train_ratio=0.9, val_ratio=0.1
    )

    train_dataloader = create_dataloader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=4, shuffle=False)

    # Use custom distillation loss
    distillation_loss_fn = CustomDistillationLoss()

    # Create optimizer with different learning rates for different layers
    optimizer = torch.optim.AdamW(
        [
            {"params": model.transformer.h.parameters(), "lr": 5e-5},
            {"params": model.lm_head.parameters(), "lr": 1e-4},
        ]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    # Training loop with advanced monitoring
    num_epochs = 3
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train with monitoring
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            distillation_loss_fn,
            epoch,
            device,
        )

        val_loss = evaluate_model(model, val_dataloader, distillation_loss_fn, device)

        # Log metrics to monitor
        monitor.log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        logger.info(f"Epoch {epoch + 1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    # Stop monitoring and generate report
    monitor.stop_monitoring()
    monitor.generate_report()

    return model, tokenizer


def model_optimization_example(model):
    """Demonstrate model optimization techniques."""

    logger = logging.getLogger(__name__)
    logger.info("=== Model Optimization Example ===")

    # Create optimizer
    optimizer = ModelOptimizer(model)

    # 1. Dynamic quantization
    logger.info("Applying dynamic quantization...")
    quantized_model = optimizer.quantize_model(quantization_type="dynamic")

    # 2. Magnitude pruning
    logger.info("Applying magnitude pruning...")
    pruned_model = optimizer.prune_model(pruning_ratio=0.1, pruning_type="magnitude")

    # 3. Inference optimization
    logger.info("Optimizing for inference...")
    optimized_model = optimizer.optimize_for_inference()

    # 4. Export to ONNX
    logger.info("Exporting to ONNX...")
    onnx_path = "optimized_model.onnx"
    optimizer.export_to_onnx(input_shape=(1, 256), output_path=onnx_path)

    # 5. Compare models
    logger.info("Comparing model performance...")
    comparison = optimizer.compare_models(optimized_model)

    logger.info(f"Original model size: {comparison['model1']['model_size_mb']:.2f} MB")
    logger.info(f"Optimized model size: {comparison['model2']['model_size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {comparison['compression_ratio']:.2f}x")

    return optimized_model


def deployment_example(model, tokenizer):
    """Demonstrate model deployment."""

    logger = logging.getLogger(__name__)
    logger.info("=== Model Deployment Example ===")

    # Save model artifacts
    from model_utils import save_model_artifacts

    model_dir = "./deployed_model"
    os.makedirs(model_dir, exist_ok=True)

    save_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        config={"model_type": "student_model"},
        output_dir=model_dir,
        model_name="deployed_model",
    )

    # Create model server
    model_server = ModelServer(
        model_path=model_dir, tokenizer_path=model_dir, device="auto", max_batch_size=4
    )

    # Load model
    model_server.load_model()

    # Test generation
    test_prompts = [
        "Write a Python function to sort a list.",
        "Explain quantum computing in simple terms.",
    ]

    logger.info("Testing model generation...")
    for prompt in test_prompts:
        generated_texts = model_server.generate_text(
            prompt=prompt, max_length=128, temperature=0.7
        )
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated_texts[0]}")
        logger.info("-" * 50)

    # Create FastAPI server (commented out to avoid blocking)
    # fastapi_server = FastAPIModelServer(model_server)
    # logger.info("Starting FastAPI server on http://localhost:8000")
    # fastapi_server.run(host='0.0.0.0', port=8000)

    return model_server


def performance_comparison_example():
    """Demonstrate performance comparison between different models."""

    logger = logging.getLogger(__name__)
    logger.info("=== Performance Comparison Example ===")

    config = ConfigManager("config.yaml")
    paths = config.get_output_paths()

    # Prepare test data
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("model.student_model_id"))

    dataset = prepare_student_dataset(
        teacher_logits_path=paths["logits_dir"], tokenizer=tokenizer, max_seq_length=256
    )

    test_dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    # Test prompts
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers.",
        "Explain the concept of machine learning.",
        "What are the benefits of renewable energy?",
        "How does a neural network work?",
        "Describe the process of photosynthesis.",
    ]

    # Compare different models
    models_to_compare = [
        ("Student Model", paths["final_model_dir"]),
        # Add more models here if available
    ]

    results = {}

    for model_name, model_path in models_to_compare:
        if os.path.exists(model_path):
            logger.info(f"Evaluating {model_name}...")

            # Create evaluator
            evaluator = ModelEvaluator(
                model_path=model_path, tokenizer_path=model_path, device="auto"
            )

            # Evaluate
            evaluation_results = evaluator.comprehensive_evaluation(
                dataloader=test_dataloader, test_prompts=test_prompts, max_samples=50
            )

            results[model_name] = evaluation_results

            logger.info(
                f"{model_name} - Perplexity: {evaluation_results['perplexity']['perplexity']:.4f}"
            )
            logger.info(
                f"{model_name} - Model Size: {evaluation_results['memory']['model_size_mb']:.2f} MB"
            )

    # Save comparison results
    import json

    comparison_dir = os.path.join(paths["base_dir"], "comparison_results")
    os.makedirs(comparison_dir, exist_ok=True)

    with open(os.path.join(comparison_dir, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Performance comparison completed")
    return results


def visualization_example():
    """Demonstrate result visualization."""

    logger = logging.getLogger(__name__)
    logger.info("=== Visualization Example ===")

    config = ConfigManager("config.yaml")
    paths = config.get_output_paths()

    # Create visualizer
    visualizer = ResultsVisualizer(
        results_dir=paths["base_dir"],
        output_dir=os.path.join(paths["base_dir"], "visualizations"),
    )

    # Generate all visualizations
    visualizer.visualize_all()

    logger.info("Visualizations generated")

    # Create interactive dashboard
    all_results = {
        "training": visualizer.load_training_results(),
        "evaluation": visualizer.load_evaluation_results(),
        "benchmark": visualizer.load_benchmark_results(),
    }

    visualizer.create_interactive_dashboard(all_results)
    logger.info("Interactive dashboard created")


def real_time_monitoring_example():
    """Demonstrate real-time monitoring."""

    logger = logging.getLogger(__name__)
    logger.info("=== Real-time Monitoring Example ===")

    config = ConfigManager("config.yaml")
    paths = config.get_output_paths()

    # Create real-time monitor
    monitor = RealTimeMonitor(os.path.join(paths["base_dir"], "training_logs"))

    # Add update callback
    def update_callback(metrics):
        if "train_loss" in metrics:
            logger.info(f"Real-time update - Train Loss: {metrics['train_loss']:.4f}")
        if "val_loss" in metrics:
            logger.info(f"Real-time update - Val Loss: {metrics['val_loss']:.4f}")

    monitor.add_update_callback(update_callback)

    # Start monitoring (this would run in background)
    # monitor.start()

    logger.info("Real-time monitoring setup completed")


def main():
    """Main function for advanced usage example."""

    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=== SteveAI Advanced Usage Example ===")

    try:
        # 1. Advanced training
        model, tokenizer = advanced_training_example()

        # 2. Model optimization
        optimized_model = model_optimization_example(model)

        # 3. Model deployment
        model_server = deployment_example(model, tokenizer)

        # 4. Performance comparison
        comparison_results = performance_comparison_example()

        # 5. Visualization
        visualization_example()

        # 6. Real-time monitoring
        real_time_monitoring_example()

        logger.info("=== Advanced Usage Example Completed ===")

    except Exception as e:
        logger.error(f"Advanced usage example failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        print_memory_usage("Final cleanup")


if __name__ == "__main__":
    main()
