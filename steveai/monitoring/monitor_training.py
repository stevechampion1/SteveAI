# SteveAI - Training Monitoring Script

import argparse
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from ..utils.config_manager import ConfigManager

# Import our custom modules
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Training monitoring and visualization class."""

    def __init__(
        self, log_dir: str, update_interval: int = 30, save_plots: bool = True
    ):
        """
        Initialize the training monitor.

        Args:
            log_dir: Directory to save monitoring data
            update_interval: Update interval in seconds
            save_plots: Whether to save plots
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.update_interval = update_interval
        self.save_plots = save_plots

        self.metrics_history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_distillation_loss": [],
            "train_hard_loss": [],
            "val_distillation_loss": [],
            "val_hard_loss": [],
            "learning_rate": [],
            "timestamp": [],
        }

        self.system_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "gpu_memory": [],
            "timestamp": [],
        }

        self.monitoring = False
        self.monitor_thread = None

        logger.info(f"Training monitor initialized with log directory: {self.log_dir}")

    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        logger.info("System monitoring stopped")

    def _monitor_system(self):
        """Monitor system resources in background."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent

                # GPU usage
                gpu_usage = 0.0
                gpu_memory = 0.0

                if torch.cuda.is_available():
                    try:
                        gpu_usage = GPUtil.getGPUs()[0].load * 100
                        gpu_memory = GPUtil.getGPUs()[0].memoryUtil * 100
                    except:
                        # Fallback to torch if GPUtil fails
                        gpu_memory = (
                            torch.cuda.memory_allocated()
                            / torch.cuda.max_memory_allocated()
                            * 100
                        )

                # Store metrics
                timestamp = time.time()
                self.system_metrics["cpu_usage"].append(cpu_usage)
                self.system_metrics["memory_usage"].append(memory_usage)
                self.system_metrics["gpu_usage"].append(gpu_usage)
                self.system_metrics["gpu_memory"].append(gpu_memory)
                self.system_metrics["timestamp"].append(timestamp)

                # Keep only recent data (last 1000 points)
                max_points = 1000
                for key in self.system_metrics:
                    if len(self.system_metrics[key]) > max_points:
                        self.system_metrics[key] = self.system_metrics[key][
                            -max_points:
                        ]

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.update_interval)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_dist_loss: float = None,
        train_hard_loss: float = None,
        val_dist_loss: float = None,
        val_hard_loss: float = None,
        learning_rate: float = None,
    ):
        """
        Log epoch metrics.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            train_dist_loss: Training distillation loss
            train_hard_loss: Training hard loss
            val_dist_loss: Validation distillation loss
            val_hard_loss: Validation hard loss
            learning_rate: Current learning rate
        """
        timestamp = time.time()

        self.metrics_history["epoch"].append(epoch)
        self.metrics_history["train_loss"].append(train_loss)
        self.metrics_history["val_loss"].append(val_loss)
        self.metrics_history["timestamp"].append(timestamp)

        if train_dist_loss is not None:
            self.metrics_history["train_distillation_loss"].append(train_dist_loss)
        if train_hard_loss is not None:
            self.metrics_history["train_hard_loss"].append(train_hard_loss)
        if val_dist_loss is not None:
            self.metrics_history["val_distillation_loss"].append(val_dist_loss)
        if val_hard_loss is not None:
            self.metrics_history["val_hard_loss"].append(val_hard_loss)
        if learning_rate is not None:
            self.metrics_history["learning_rate"].append(learning_rate)

        logger.info(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.log_dir / "training_metrics.json"

        # Convert timestamps to readable format
        readable_metrics = {}
        for key, values in self.metrics_history.items():
            if key == "timestamp":
                readable_metrics[key] = [
                    datetime.fromtimestamp(ts).isoformat() for ts in values
                ]
            else:
                readable_metrics[key] = values

        with open(metrics_file, "w") as f:
            json.dump(readable_metrics, f, indent=2)

        logger.info(f"Metrics saved to: {metrics_file}")

    def save_system_metrics(self):
        """Save system metrics to JSON file."""
        system_file = self.log_dir / "system_metrics.json"

        # Convert timestamps to readable format
        readable_system = {}
        for key, values in self.system_metrics.items():
            if key == "timestamp":
                readable_system[key] = [
                    datetime.fromtimestamp(ts).isoformat() for ts in values
                ]
            else:
                readable_system[key] = values

        with open(system_file, "w") as f:
            json.dump(readable_system, f, indent=2)

        logger.info(f"System metrics saved to: {system_file}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        if not self.metrics_history["epoch"]:
            logger.warning("No training metrics to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress", fontsize=16)

        epochs = self.metrics_history["epoch"]

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(
            epochs, self.metrics_history["train_loss"], label="Train Loss", color="blue"
        )
        ax1.plot(
            epochs, self.metrics_history["val_loss"], label="Val Loss", color="red"
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Distillation vs Hard Loss
        ax2 = axes[0, 1]
        if self.metrics_history["train_distillation_loss"]:
            ax2.plot(
                epochs,
                self.metrics_history["train_distillation_loss"],
                label="Train Distillation Loss",
                color="green",
            )
        if self.metrics_history["train_hard_loss"]:
            ax2.plot(
                epochs,
                self.metrics_history["train_hard_loss"],
                label="Train Hard Loss",
                color="orange",
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Distillation vs Hard Loss")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Learning rate
        ax3 = axes[1, 0]
        if self.metrics_history["learning_rate"]:
            ax3.plot(epochs, self.metrics_history["learning_rate"], color="purple")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate Schedule")
            ax3.grid(True)

        # Plot 4: Validation losses
        ax4 = axes[1, 1]
        if self.metrics_history["val_distillation_loss"]:
            ax4.plot(
                epochs,
                self.metrics_history["val_distillation_loss"],
                label="Val Distillation Loss",
                color="green",
            )
        if self.metrics_history["val_hard_loss"]:
            ax4.plot(
                epochs,
                self.metrics_history["val_hard_loss"],
                label="Val Hard Loss",
                color="orange",
            )
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss")
        ax4.set_title("Validation Losses")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        # Save plot
        if self.save_plots:
            plot_file = self.log_dir / "training_curves.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            logger.info(f"Training curves saved to: {plot_file}")

        plt.close()

    def plot_system_metrics(self):
        """Plot and save system metrics."""
        if not self.system_metrics["timestamp"]:
            logger.warning("No system metrics to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("System Resource Usage", fontsize=16)

        timestamps = self.system_metrics["timestamp"]
        time_labels = [
            datetime.fromtimestamp(ts).strftime("%H:%M:%S") for ts in timestamps
        ]

        # Plot 1: CPU Usage
        ax1 = axes[0, 0]
        ax1.plot(time_labels, self.system_metrics["cpu_usage"], color="blue")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("CPU Usage (%)")
        ax1.set_title("CPU Usage")
        ax1.grid(True)
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Memory Usage
        ax2 = axes[0, 1]
        ax2.plot(time_labels, self.system_metrics["memory_usage"], color="red")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Memory Usage (%)")
        ax2.set_title("Memory Usage")
        ax2.grid(True)
        ax2.tick_params(axis="x", rotation=45)

        # Plot 3: GPU Usage
        ax3 = axes[1, 0]
        ax3.plot(time_labels, self.system_metrics["gpu_usage"], color="green")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("GPU Usage (%)")
        ax3.set_title("GPU Usage")
        ax3.grid(True)
        ax3.tick_params(axis="x", rotation=45)

        # Plot 4: GPU Memory
        ax4 = axes[1, 1]
        ax4.plot(time_labels, self.system_metrics["gpu_memory"], color="orange")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("GPU Memory Usage (%)")
        ax4.set_title("GPU Memory Usage")
        ax4.grid(True)
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        if self.save_plots:
            plot_file = self.log_dir / "system_metrics.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            logger.info(f"System metrics plot saved to: {plot_file}")

        plt.close()

    def generate_report(self):
        """Generate training report."""
        if not self.metrics_history["epoch"]:
            logger.warning("No training metrics to generate report")
            return

        report = {
            "training_summary": {
                "total_epochs": len(self.metrics_history["epoch"]),
                "final_train_loss": self.metrics_history["train_loss"][-1],
                "final_val_loss": self.metrics_history["val_loss"][-1],
                "best_val_loss": min(self.metrics_history["val_loss"]),
                "best_epoch": self.metrics_history["epoch"][
                    self.metrics_history["val_loss"].index(
                        min(self.metrics_history["val_loss"])
                    )
                ],
            }
        }

        # Add system metrics summary
        if self.system_metrics["timestamp"]:
            report["system_summary"] = {
                "avg_cpu_usage": np.mean(self.system_metrics["cpu_usage"]),
                "max_cpu_usage": max(self.system_metrics["cpu_usage"]),
                "avg_memory_usage": np.mean(self.system_metrics["memory_usage"]),
                "max_memory_usage": max(self.system_metrics["memory_usage"]),
                "avg_gpu_usage": np.mean(self.system_metrics["gpu_usage"]),
                "max_gpu_usage": max(self.system_metrics["gpu_usage"]),
                "avg_gpu_memory": np.mean(self.system_metrics["gpu_memory"]),
                "max_gpu_memory": max(self.system_metrics["gpu_memory"]),
            }

        # Save report
        report_file = self.log_dir / "training_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved to: {report_file}")
        return report

    def cleanup(self):
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        self.save_metrics()
        self.save_system_metrics()
        self.plot_training_curves()
        self.plot_system_metrics()
        self.generate_report()

        logger.info("Training monitoring cleanup completed")


class RealTimeMonitor:
    """Real-time monitoring with live updates."""

    def __init__(self, log_dir: str):
        """
        Initialize real-time monitor.

        Args:
            log_dir: Directory to save monitoring data
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = TrainingMonitor(log_dir)
        self.update_callbacks = []

        logger.info("Real-time monitor initialized")

    def add_update_callback(self, callback: Callable):
        """
        Add callback function for updates.

        Args:
            callback: Function to call on updates
        """
        self.update_callbacks.append(callback)

    def start(self):
        """Start real-time monitoring."""
        self.monitor.start_monitoring()

        # Start update loop
        def update_loop():
            while self.monitor.monitoring:
                # Call update callbacks
                for callback in self.update_callbacks:
                    try:
                        callback(self.monitor)
                    except Exception as e:
                        logger.error(f"Error in update callback: {e}")

                time.sleep(10)  # Update every 10 seconds

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

        logger.info("Real-time monitoring started")

    def stop(self):
        """Stop real-time monitoring."""
        self.monitor.cleanup()
        logger.info("Real-time monitoring stopped")


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Monitor SteveAI training")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./training_logs",
        help="Directory to save monitoring data",
    )
    parser.add_argument(
        "--update_interval", type=int, default=30, help="Update interval in seconds"
    )
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    parser.add_argument(
        "--real_time", action="store_true", help="Enable real-time monitoring"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Training Monitor ---")

        if args.real_time:
            # Real-time monitoring
            monitor = RealTimeMonitor(args.log_dir)
            monitor.start()

            try:
                # Keep running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping real-time monitoring...")
                monitor.stop()
        else:
            # Regular monitoring
            monitor = TrainingMonitor(
                log_dir=args.log_dir,
                update_interval=args.update_interval,
                save_plots=args.save_plots,
            )

            # Example usage
            monitor.start_monitoring()

            # Simulate some training
            for epoch in range(1, 6):
                train_loss = 2.0 - epoch * 0.3 + np.random.normal(0, 0.1)
                val_loss = 2.2 - epoch * 0.25 + np.random.normal(0, 0.1)

                monitor.log_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_dist_loss=train_loss * 0.7,
                    train_hard_loss=train_loss * 0.3,
                    learning_rate=5e-5 * (0.9**epoch),
                )

                time.sleep(2)

            monitor.cleanup()

        logger.info("--- Training Monitor Completed ---")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
