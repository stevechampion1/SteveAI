# SteveAI - Results Visualization Script

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
from plotly.subplots import make_subplots

from ..utils.config_manager import ConfigManager

# Import our custom modules
from ..utils.utils import load_json, save_json, setup_logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Results visualization and analysis class."""

    def __init__(self, results_dir: str, output_dir: str = "./visualizations"):
        """
        Initialize the results visualizer.

        Args:
            results_dir: Directory containing results files
            output_dir: Directory to save visualizations
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        logger.info(f"Results visualizer initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_training_results(self) -> Dict[str, Any]:
        """Load training results from JSON files."""
        results = {}

        # Load training metrics
        training_metrics_file = self.results_dir / "training_metrics.json"
        if training_metrics_file.exists():
            results["training_metrics"] = load_json(str(training_metrics_file))

        # Load system metrics
        system_metrics_file = self.results_dir / "system_metrics.json"
        if system_metrics_file.exists():
            results["system_metrics"] = load_json(str(system_metrics_file))

        # Load training report
        training_report_file = self.results_dir / "training_report.json"
        if training_report_file.exists():
            results["training_report"] = load_json(str(training_report_file))

        return results

    def load_evaluation_results(self) -> Dict[str, Any]:
        """Load evaluation results from JSON files."""
        results = {}

        # Load evaluation results
        eval_results_file = self.results_dir / "evaluation_results.json"
        if eval_results_file.exists():
            results["evaluation"] = load_json(str(eval_results_file))

        return results

    def load_benchmark_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON files."""
        results = {}

        # Load benchmark results
        benchmark_results_file = self.results_dir / "benchmark_results.json"
        if benchmark_results_file.exists():
            results["benchmark"] = load_json(str(benchmark_results_file))

        return results

    def plot_training_curves(self, training_metrics: Dict[str, Any]) -> None:
        """Plot training curves."""
        if not training_metrics:
            logger.warning("No training metrics to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

        epochs = training_metrics.get("epoch", [])
        if not epochs:
            logger.warning("No epoch data found")
            return

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if "train_loss" in training_metrics:
            ax1.plot(
                epochs,
                training_metrics["train_loss"],
                label="Train Loss",
                color="blue",
                linewidth=2,
            )
        if "val_loss" in training_metrics:
            ax1.plot(
                epochs,
                training_metrics["val_loss"],
                label="Val Loss",
                color="red",
                linewidth=2,
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distillation vs Hard Loss
        ax2 = axes[0, 1]
        if "train_distillation_loss" in training_metrics:
            ax2.plot(
                epochs,
                training_metrics["train_distillation_loss"],
                label="Train Distillation Loss",
                color="green",
                linewidth=2,
            )
        if "train_hard_loss" in training_metrics:
            ax2.plot(
                epochs,
                training_metrics["train_hard_loss"],
                label="Train Hard Loss",
                color="orange",
                linewidth=2,
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Distillation vs Hard Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Learning rate
        ax3 = axes[1, 0]
        if "learning_rate" in training_metrics:
            ax3.plot(
                epochs, training_metrics["learning_rate"], color="purple", linewidth=2
            )
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate Schedule")
            ax3.grid(True, alpha=0.3)

        # Plot 4: Validation losses
        ax4 = axes[1, 1]
        if "val_distillation_loss" in training_metrics:
            ax4.plot(
                epochs,
                training_metrics["val_distillation_loss"],
                label="Val Distillation Loss",
                color="green",
                linewidth=2,
            )
        if "val_hard_loss" in training_metrics:
            ax4.plot(
                epochs,
                training_metrics["val_hard_loss"],
                label="Val Hard Loss",
                color="orange",
                linewidth=2,
            )
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss")
        ax4.set_title("Validation Losses")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        logger.info(f"Training curves saved to: {plot_file}")

        plt.close()

    def plot_system_metrics(self, system_metrics: Dict[str, Any]) -> None:
        """Plot system metrics."""
        if not system_metrics:
            logger.warning("No system metrics to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("System Resource Usage", fontsize=16, fontweight="bold")

        timestamps = system_metrics.get("timestamp", [])
        if not timestamps:
            logger.warning("No timestamp data found")
            return

        # Convert timestamps to datetime objects
        time_labels = [datetime.fromisoformat(ts) for ts in timestamps]

        # Plot 1: CPU Usage
        ax1 = axes[0, 0]
        if "cpu_usage" in system_metrics:
            ax1.plot(
                time_labels, system_metrics["cpu_usage"], color="blue", linewidth=2
            )
            ax1.set_xlabel("Time")
            ax1.set_ylabel("CPU Usage (%)")
            ax1.set_title("CPU Usage")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Memory Usage
        ax2 = axes[0, 1]
        if "memory_usage" in system_metrics:
            ax2.plot(
                time_labels, system_metrics["memory_usage"], color="red", linewidth=2
            )
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Memory Usage (%)")
            ax2.set_title("Memory Usage")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

        # Plot 3: GPU Usage
        ax3 = axes[1, 0]
        if "gpu_usage" in system_metrics:
            ax3.plot(
                time_labels, system_metrics["gpu_usage"], color="green", linewidth=2
            )
            ax3.set_xlabel("Time")
            ax3.set_ylabel("GPU Usage (%)")
            ax3.set_title("GPU Usage")
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis="x", rotation=45)

        # Plot 4: GPU Memory
        ax4 = axes[1, 1]
        if "gpu_memory" in system_metrics:
            ax4.plot(
                time_labels, system_metrics["gpu_memory"], color="orange", linewidth=2
            )
            ax4.set_xlabel("Time")
            ax4.set_ylabel("GPU Memory Usage (%)")
            ax4.set_title("GPU Memory Usage")
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "system_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        logger.info(f"System metrics plot saved to: {plot_file}")

        plt.close()

    def plot_evaluation_results(self, evaluation_results: Dict[str, Any]) -> None:
        """Plot evaluation results."""
        if not evaluation_results:
            logger.warning("No evaluation results to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

        # Plot 1: Perplexity
        ax1 = axes[0, 0]
        if "perplexity" in evaluation_results:
            perplexity = evaluation_results["perplexity"]["perplexity"]
            ax1.bar(["Perplexity"], [perplexity], color="blue", alpha=0.7)
            ax1.set_ylabel("Perplexity")
            ax1.set_title("Model Perplexity")
            ax1.grid(True, alpha=0.3)

        # Plot 2: Memory Usage
        ax2 = axes[0, 1]
        if "memory" in evaluation_results:
            memory = evaluation_results["memory"]
            metrics = ["Total Parameters", "GPU Memory (GB)", "CPU Memory (GB)"]
            values = [
                memory["total_parameters"] / 1e6,  # Convert to millions
                memory["gpu_memory_allocated_gb"],
                memory["cpu_memory_gb"],
            ]
            ax2.bar(metrics, values, color=["green", "orange", "red"], alpha=0.7)
            ax2.set_ylabel("Value")
            ax2.set_title("Memory Usage")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

        # Plot 3: Generation Quality
        ax3 = axes[1, 0]
        if "generation" in evaluation_results:
            generation = evaluation_results["generation"]
            metrics = ["Avg Generation Time (s)", "Avg Text Length (words)"]
            values = [generation["avg_generation_time"], generation["avg_text_length"]]
            ax3.bar(metrics, values, color=["purple", "brown"], alpha=0.7)
            ax3.set_ylabel("Value")
            ax3.set_title("Generation Quality")
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis="x", rotation=45)

        # Plot 4: Teacher Comparison
        ax4 = axes[1, 1]
        if "teacher_comparison" in evaluation_results:
            comparison = evaluation_results["teacher_comparison"]
            metrics = ["Loss Ratio", "Logit Similarity"]
            values = [comparison["loss_ratio"], comparison["avg_logit_similarity"]]
            ax4.bar(metrics, values, color=["cyan", "magenta"], alpha=0.7)
            ax4.set_ylabel("Value")
            ax4.set_title("Teacher Comparison")
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "evaluation_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        logger.info(f"Evaluation results plot saved to: {plot_file}")

        plt.close()

    def plot_benchmark_results(self, benchmark_results: Dict[str, Any]) -> None:
        """Plot benchmark results."""
        if not benchmark_results:
            logger.warning("No benchmark results to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Benchmark Results", fontsize=16, fontweight="bold")

        # Plot 1: Model Size
        ax1 = axes[0, 0]
        if "model_size" in benchmark_results:
            model_size = benchmark_results["model_size"]
            metrics = ["Total Parameters (M)", "Model Size (MB)"]
            values = [
                model_size["total_parameters"] / 1e6,  # Convert to millions
                model_size["model_size_mb"],
            ]
            ax1.bar(metrics, values, color=["blue", "green"], alpha=0.7)
            ax1.set_ylabel("Value")
            ax1.set_title("Model Size")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Inference Speed
        ax2 = axes[0, 1]
        if "inference_speed" in benchmark_results:
            inference = benchmark_results["inference_speed"]
            metrics = ["Avg Inference Time (s)", "Avg Tokens/sec"]
            values = [
                inference["avg_inference_time"],
                inference["avg_tokens_per_second"],
            ]
            ax2.bar(metrics, values, color=["red", "orange"], alpha=0.7)
            ax2.set_ylabel("Value")
            ax2.set_title("Inference Speed")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

        # Plot 3: Memory Usage
        ax3 = axes[1, 0]
        if "memory_usage" in benchmark_results:
            memory = benchmark_results["memory_usage"]
            metrics = ["Peak GPU Memory (GB)", "Avg CPU Memory (GB)"]
            values = [memory["gpu_memory_peak_gb"], memory["avg_cpu_memory_gb"]]
            ax3.bar(metrics, values, color=["purple", "brown"], alpha=0.7)
            ax3.set_ylabel("Value")
            ax3.set_title("Memory Usage")
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis="x", rotation=45)

        # Plot 4: Model Comparison
        ax4 = axes[1, 1]
        if "model_comparison" in benchmark_results:
            comparison = benchmark_results["model_comparison"]
            metrics = ["Speedup", "Compression Ratio"]
            values = [comparison["speedup"], comparison["compression_ratio"]]
            ax4.bar(metrics, values, color=["cyan", "magenta"], alpha=0.7)
            ax4.set_ylabel("Value")
            ax4.set_title("Model Comparison")
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "benchmark_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        logger.info(f"Benchmark results plot saved to: {plot_file}")

        plt.close()

    def create_interactive_dashboard(self, all_results: Dict[str, Any]) -> None:
        """Create interactive dashboard using Plotly."""
        if not all_results:
            logger.warning("No results to create dashboard")
            return

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Training Progress",
                "System Metrics",
                "Evaluation Results",
                "Benchmark Results",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Plot 1: Training Progress
        if "training_metrics" in all_results:
            training_metrics = all_results["training_metrics"]
            epochs = training_metrics.get("epoch", [])

            if "train_loss" in training_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=training_metrics["train_loss"],
                        mode="lines",
                        name="Train Loss",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

            if "val_loss" in training_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=training_metrics["val_loss"],
                        mode="lines",
                        name="Val Loss",
                        line=dict(color="red"),
                    ),
                    row=1,
                    col=1,
                )

        # Plot 2: System Metrics
        if "system_metrics" in all_results:
            system_metrics = all_results["system_metrics"]
            timestamps = system_metrics.get("timestamp", [])

            if "cpu_usage" in system_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=system_metrics["cpu_usage"],
                        mode="lines",
                        name="CPU Usage",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=2,
                )

            if "gpu_usage" in system_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=system_metrics["gpu_usage"],
                        mode="lines",
                        name="GPU Usage",
                        line=dict(color="orange"),
                    ),
                    row=1,
                    col=2,
                )

        # Plot 3: Evaluation Results
        if "evaluation" in all_results:
            evaluation = all_results["evaluation"]

            if "perplexity" in evaluation:
                perplexity = evaluation["perplexity"]["perplexity"]
                fig.add_trace(
                    go.Bar(
                        x=["Perplexity"],
                        y=[perplexity],
                        name="Perplexity",
                        marker_color="purple",
                    ),
                    row=2,
                    col=1,
                )

        # Plot 4: Benchmark Results
        if "benchmark" in all_results:
            benchmark = all_results["benchmark"]

            if "model_size" in benchmark:
                model_size = benchmark["model_size"]
                fig.add_trace(
                    go.Bar(
                        x=["Total Parameters (M)", "Model Size (MB)"],
                        y=[
                            model_size["total_parameters"] / 1e6,
                            model_size["model_size_mb"],
                        ],
                        name="Model Size",
                        marker_color="cyan",
                    ),
                    row=2,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title_text="SteveAI Model Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800,
            width=1200,
        )

        # Save interactive dashboard
        dashboard_file = self.output_dir / "interactive_dashboard.html"
        pyo.plot(fig, filename=str(dashboard_file), auto_open=False)
        logger.info(f"Interactive dashboard saved to: {dashboard_file}")

    def generate_summary_report(self, all_results: Dict[str, Any]) -> None:
        """Generate summary report."""
        if not all_results:
            logger.warning("No results to generate report")
            return

        report = {"generated_at": datetime.now().isoformat(), "summary": {}}

        # Training summary
        if "training_metrics" in all_results:
            training_metrics = all_results["training_metrics"]
            if "train_loss" in training_metrics and "val_loss" in training_metrics:
                report["summary"]["training"] = {
                    "final_train_loss": training_metrics["train_loss"][-1],
                    "final_val_loss": training_metrics["val_loss"][-1],
                    "best_val_loss": min(training_metrics["val_loss"]),
                    "total_epochs": len(training_metrics["epoch"]),
                }

        # Evaluation summary
        if "evaluation" in all_results:
            evaluation = all_results["evaluation"]
            if "perplexity" in evaluation:
                report["summary"]["evaluation"] = {
                    "perplexity": evaluation["perplexity"]["perplexity"]
                }

        # Benchmark summary
        if "benchmark" in all_results:
            benchmark = all_results["benchmark"]
            if "model_size" in benchmark:
                report["summary"]["benchmark"] = {
                    "total_parameters": benchmark["model_size"]["total_parameters"],
                    "model_size_mb": benchmark["model_size"]["model_size_mb"],
                }

        # Save report
        report_file = self.output_dir / "summary_report.json"
        save_json(report, str(report_file))
        logger.info(f"Summary report saved to: {report_file}")

    def visualize_all(self) -> None:
        """Visualize all available results."""
        logger.info("Starting comprehensive visualization...")

        # Load all results
        training_results = self.load_training_results()
        evaluation_results = self.load_evaluation_results()
        benchmark_results = self.load_benchmark_results()

        # Combine all results
        all_results = {**training_results, **evaluation_results, **benchmark_results}

        # Create visualizations
        if "training_metrics" in all_results:
            self.plot_training_curves(all_results["training_metrics"])

        if "system_metrics" in all_results:
            self.plot_system_metrics(all_results["system_metrics"])

        if "evaluation" in all_results:
            self.plot_evaluation_results(all_results["evaluation"])

        if "benchmark" in all_results:
            self.plot_benchmark_results(all_results["benchmark"])

        # Create interactive dashboard
        self.create_interactive_dashboard(all_results)

        # Generate summary report
        self.generate_summary_report(all_results)

        logger.info("Comprehensive visualization completed")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize SteveAI results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing results files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Create interactive dashboard"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Results Visualization ---")

        # Initialize visualizer
        visualizer = ResultsVisualizer(
            results_dir=args.results_dir, output_dir=args.output_dir
        )

        # Create visualizations
        visualizer.visualize_all()

        logger.info("--- Results Visualization Completed ---")

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
