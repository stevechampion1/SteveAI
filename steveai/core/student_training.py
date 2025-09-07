# SteveAI - Student Model Training Script

import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.data_collator import default_data_collator

from ..utils.config_manager import ConfigManager
from ..utils.model_utils import load_model_checkpoint, save_model_checkpoint
from ..utils.utils import check_disk_space, print_memory_usage, setup_logging
from .data_utils import load_teacher_logits, prepare_student_dataset

# Import our custom modules
from .distillation_loss import DistillationLoss


class StudentTrainingConfig:
    """Configuration class for student model training."""

    # Model Configuration
    STUDENT_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
    TOKENIZER_PATH = STUDENT_MODEL_ID

    # Training Configuration
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    BATCH_SIZE = 8  # Can be larger than teacher inference
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_GRAD_NORM = 1.0
    WARMUP_STEPS = 100

    # Distillation Configuration
    TEMPERATURE = 4.0
    ALPHA = 0.7  # Weight for distillation loss vs hard loss
    BETA = 0.3  # Weight for hard loss

    # Data Configuration
    MAX_SEQ_LENGTH = 256
    DATALOADER_NUM_WORKERS = 2

    # Output Configuration
    @staticmethod
    def get_output_base_dir() -> str:
        """Get output directory based on environment."""
        if os.path.exists("/kaggle/working"):
            return "/kaggle/working/SteveAI_Student_Training"
        else:
            return "./output/SteveAI_Student_Training"

    @classmethod
    def get_checkpoint_dir(cls) -> str:
        """Get checkpoint directory."""
        return os.path.join(cls.get_output_base_dir(), "checkpoints")

    @classmethod
    def get_final_model_dir(cls) -> str:
        """Get final model directory."""
        return os.path.join(cls.get_output_base_dir(), "final_model")


# Create global config instance
config = StudentTrainingConfig()


def validate_training_config() -> None:
    """Validate training configuration parameters."""
    if config.LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive")

    if config.NUM_EPOCHS <= 0:
        raise ValueError("NUM_EPOCHS must be positive")

    if config.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")

    if config.TEMPERATURE <= 0:
        raise ValueError("TEMPERATURE must be positive")

    if not (0 <= config.ALPHA <= 1):
        raise ValueError("ALPHA must be between 0 and 1")

    if not (0 <= config.BETA <= 1):
        raise ValueError("BETA must be between 0 and 1")

    if abs(config.ALPHA + config.BETA - 1.0) > 1e-6:
        raise ValueError("ALPHA + BETA must equal 1.0")

    logger = logging.getLogger(__name__)
    logger.info("Training configuration validation passed.")


def load_student_model_and_tokenizer():
    """Load student model and tokenizer."""
    logger = logging.getLogger(__name__)

    logger.info(f"Loading student model: {config.STUDENT_MODEL_ID}")
    student_model = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    logger.info(f"Loading tokenizer: {config.TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_PATH, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        logger.info("Tokenizer lacks pad_token, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return student_model, tokenizer


def create_optimizer_and_scheduler(model, num_training_steps):
    """Create optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def train_epoch(
    model, dataloader, optimizer, scheduler, distillation_loss_fn, epoch, device
):
    """Train for one epoch."""
    logger = logging.getLogger(__name__)
    model.train()

    total_loss = 0.0
    total_distillation_loss = 0.0
    total_hard_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Calculate losses
        distillation_loss, hard_loss = distillation_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=config.TEMPERATURE,
            alpha=config.ALPHA,
            beta=config.BETA,
        )

        total_loss_batch = distillation_loss + hard_loss

        # Backward pass
        total_loss_batch = total_loss_batch / config.GRADIENT_ACCUMULATION_STEPS
        total_loss_batch.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update metrics
        total_loss += total_loss_batch.item() * config.GRADIENT_ACCUMULATION_STEPS
        total_distillation_loss += distillation_loss.item()
        total_hard_loss += hard_loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{total_loss_batch.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}",
                "dist_loss": f"{distillation_loss.item():.4f}",
                "hard_loss": f"{hard_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

        # Memory cleanup
        if (batch_idx + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_memory_usage(f"Training Batch {batch_idx+1}/{num_batches}")

    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_distillation_loss = total_distillation_loss / num_batches
    avg_hard_loss = total_hard_loss / num_batches

    return avg_loss, avg_distillation_loss, avg_hard_loss


def evaluate_model(model, dataloader, distillation_loss_fn, device):
    """Evaluate the model."""
    logger = logging.getLogger(__name__)
    model.eval()

    total_loss = 0.0
    total_distillation_loss = 0.0
    total_hard_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Calculate losses
            distillation_loss, hard_loss = distillation_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=config.TEMPERATURE,
                alpha=config.ALPHA,
                beta=config.BETA,
            )

            total_loss_batch = distillation_loss + hard_loss

            # Update metrics
            total_loss += total_loss_batch.item()
            total_distillation_loss += distillation_loss.item()
            total_hard_loss += hard_loss.item()

    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_distillation_loss = total_distillation_loss / num_batches
    avg_hard_loss = total_hard_loss / num_batches

    return avg_loss, avg_distillation_loss, avg_hard_loss


def main():
    """Main training function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting SteveAI Student Model Training ---")

        # Validate configuration
        validate_training_config()

        # Create output directories
        os.makedirs(config.get_checkpoint_dir(), exist_ok=True)
        os.makedirs(config.get_final_model_dir(), exist_ok=True)

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        print_memory_usage("Initial")
        check_disk_space()

        # Load student model and tokenizer
        student_model, tokenizer = load_student_model_and_tokenizer()
        student_model = student_model.to(device)

        # Load teacher logits and prepare dataset
        logger.info("Loading teacher logits and preparing dataset...")
        dataset = prepare_student_dataset(
            teacher_logits_path="../teacher_inference_gpu.py",  # This should be the actual path
            tokenizer=tokenizer,
            max_seq_length=config.MAX_SEQ_LENGTH,
        )

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        # Create distillation loss function
        distillation_loss_fn = DistillationLoss()

        # Calculate training steps
        num_training_steps = (
            len(train_dataloader)
            * config.NUM_EPOCHS
            // config.GRADIENT_ACCUMULATION_STEPS
        )

        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            student_model, num_training_steps
        )

        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {config.NUM_EPOCHS}")
        logger.info(f"  - Batch size: {config.BATCH_SIZE}")
        logger.info(f"  - Learning rate: {config.LEARNING_RATE}")
        logger.info(f"  - Temperature: {config.TEMPERATURE}")
        logger.info(f"  - Alpha (distillation): {config.ALPHA}")
        logger.info(f"  - Beta (hard): {config.BETA}")
        logger.info(f"  - Training steps: {num_training_steps}")

        # Training loop
        best_val_loss = float("inf")
        training_history = []

        for epoch in range(config.NUM_EPOCHS):
            logger.info(f"--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

            # Train
            train_loss, train_dist_loss, train_hard_loss = train_epoch(
                student_model,
                train_dataloader,
                optimizer,
                scheduler,
                distillation_loss_fn,
                epoch,
                device,
            )

            # Evaluate
            val_loss, val_dist_loss, val_hard_loss = evaluate_model(
                student_model, val_dataloader, distillation_loss_fn, device
            )

            # Log metrics
            logger.info(f"Epoch {epoch+1} Results:")
            logger.info(
                f"  Train Loss: {train_loss:.4f} (Dist: {train_dist_loss:.4f}, Hard: {train_hard_loss:.4f})"
            )
            logger.info(
                f"  Val Loss: {val_loss:.4f} (Dist: {val_dist_loss:.4f}, Hard: {val_hard_loss:.4f})"
            )

            # Save training history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_distillation_loss": train_dist_loss,
                    "train_hard_loss": train_hard_loss,
                    "val_loss": val_loss,
                    "val_distillation_loss": val_dist_loss,
                    "val_hard_loss": val_hard_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    config.get_checkpoint_dir(), f"best_model_epoch_{epoch+1}.pt"
                )
                save_model_checkpoint(
                    student_model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    checkpoint_path,
                )
                logger.info(f"New best model saved: {checkpoint_path}")

            # Save regular checkpoint
            if (epoch + 1) % 2 == 0:  # Save every 2 epochs
                checkpoint_path = os.path.join(
                    config.get_checkpoint_dir(), f"checkpoint_epoch_{epoch+1}.pt"
                )
                save_model_checkpoint(
                    student_model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    checkpoint_path,
                )

            print_memory_usage(f"After Epoch {epoch+1}")

        # Save final model
        final_model_path = os.path.join(config.get_final_model_dir(), "final_model")
        student_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Save training history
        history_path = os.path.join(
            config.get_output_base_dir(), "training_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        logger.info("--- Student Model Training Completed ---")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if "student_model" in locals():
            del student_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage("Final")


if __name__ == "__main__":
    main()
