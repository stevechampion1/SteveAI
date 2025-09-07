# SteveAI - Data Processing Utilities

import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TeacherLogitsLoader:
    """Utility class for loading and managing teacher logits."""

    def __init__(self, logits_dir: str):
        """
        Initialize the teacher logits loader.

        Args:
            logits_dir: Directory containing teacher logits files
        """
        self.logits_dir = Path(logits_dir)
        if not self.logits_dir.exists():
            raise FileNotFoundError(f"Teacher logits directory not found: {logits_dir}")

        self.logits_files = self._find_logits_files()
        logger.info(f"Found {len(self.logits_files)} teacher logits files")

    def _find_logits_files(self) -> List[Path]:
        """Find all teacher logits files in the directory."""
        logits_files = []
        for file_path in self.logits_dir.glob("teacher_logits_batch_*.pt"):
            logits_files.append(file_path)

        # Sort by batch number
        logits_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return logits_files

    def load_logits(self, batch_idx: int) -> torch.Tensor:
        """
        Load teacher logits for a specific batch.

        Args:
            batch_idx: Batch index

        Returns:
            Teacher logits tensor
        """
        if batch_idx >= len(self.logits_files):
            raise IndexError(
                f"Batch index {batch_idx} out of range. "
                f"Available batches: 0-{len(self.logits_files)-1}"
            )

        file_path = self.logits_files[batch_idx]
        try:
            data = torch.load(file_path, map_location="cpu")
            if "teacher_logits" in data:
                return data["teacher_logits"]
            else:
                # Handle different data formats
                return data
        except Exception as e:
            logger.error(f"Error loading logits from {file_path}: {e}")
            raise

    def load_all_logits(self) -> List[torch.Tensor]:
        """
        Load all teacher logits.

        Returns:
            List of teacher logits tensors
        """
        all_logits = []
        for i in range(len(self.logits_files)):
            logits = self.load_logits(i)
            all_logits.append(logits)

        logger.info(f"Loaded {len(all_logits)} batches of teacher logits")
        return all_logits

    def get_logits_info(self) -> Dict[str, Any]:
        """
        Get information about the teacher logits.

        Returns:
            Dictionary containing logits information
        """
        if not self.logits_files:
            return {}

        # Load first batch to get shape info
        first_logits = self.load_logits(0)

        info = {
            "num_batches": len(self.logits_files),
            "logits_shape": list(first_logits.shape),
            "dtype": str(first_logits.dtype),
            "total_size_mb": sum(f.stat().st_size for f in self.logits_files)
            / (1024 * 1024),
        }

        return info


class StudentDataset(torch.utils.data.Dataset):
    """Dataset class for student training with teacher logits."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):
        """
        Initialize the student dataset.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            teacher_logits: Teacher model logits
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.teacher_logits = teacher_logits

        # Validate shapes
        if not (
            self.input_ids.shape[0]
            == self.attention_mask.shape[0]
            == self.labels.shape[0]
            == self.teacher_logits.shape[0]
        ):
            raise ValueError("All tensors must have the same batch dimension")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "teacher_logits": self.teacher_logits[idx],
        }


def load_teacher_logits(logits_dir: str) -> List[torch.Tensor]:
    """
    Load teacher logits from directory.

    Args:
        logits_dir: Directory containing teacher logits files

    Returns:
        List of teacher logits tensors
    """
    loader = TeacherLogitsLoader(logits_dir)
    return loader.load_all_logits()


def prepare_student_dataset(
    teacher_logits_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 256,
    dataset_id: str = "yahma/alpaca-cleaned",
    dataset_subset_size: Optional[int] = None,
) -> StudentDataset:
    """
    Prepare dataset for student training.

    Args:
        teacher_logits_path: Path to teacher logits directory
        tokenizer: Tokenizer for processing text
        max_seq_length: Maximum sequence length
        dataset_id: Hugging Face dataset ID
        dataset_subset_size: Number of samples to use (None for all)

    Returns:
        StudentDataset object
    """
    logger.info("Preparing student dataset...")

    # Load teacher logits
    teacher_logits_list = load_teacher_logits(teacher_logits_path)

    # Load and process dataset
    logger.info(f"Loading dataset: {dataset_id}")
    split_string = "train"
    slice_str = f"[:{dataset_subset_size}]" if dataset_subset_size is not None else ""
    raw_dataset = load_dataset(dataset_id, split=f"{split_string}{slice_str}")

    # Create prompts
    def create_prompt(example: Dict[str, Any]) -> str:
        """Create a prompt string from an Alpaca-style dataset example."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        if input_text and input_text.strip():
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output_text}"""
        return prompt.strip()

    # Process dataset
    prompt_dataset = raw_dataset.map(
        lambda example: {"prompt_text": create_prompt(example)}, desc="Creating prompts"
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt_text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )

    tokenized_dataset = prompt_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=prompt_dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Convert to tensors
    input_ids = torch.stack([torch.tensor(x) for x in tokenized_dataset["input_ids"]])
    attention_mask = torch.stack(
        [torch.tensor(x) for x in tokenized_dataset["attention_mask"]]
    )

    # Create labels (shifted input_ids for causal LM)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Ignore last token

    # Concatenate teacher logits
    teacher_logits = torch.cat(teacher_logits_list, dim=0)

    # Ensure all tensors have the same number of samples
    min_samples = min(len(input_ids), len(teacher_logits))
    input_ids = input_ids[:min_samples]
    attention_mask = attention_mask[:min_samples]
    labels = labels[:min_samples]
    teacher_logits = teacher_logits[:min_samples]

    logger.info(f"Prepared dataset with {min_samples} samples")
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Teacher logits shape: {teacher_logits.shape}")

    return StudentDataset(input_ids, attention_mask, labels, teacher_logits)


def create_data_splits(
    dataset: StudentDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[StudentDataset, StudentDataset, StudentDataset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: StudentDataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Create random indices
    torch.manual_seed(random_seed)
    indices = torch.randperm(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    logger.info(
        f"Dataset split: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def save_dataset_info(dataset: StudentDataset, save_path: str) -> None:
    """
    Save dataset information to file.

    Args:
        dataset: StudentDataset to save info for
        save_path: Path to save the info file
    """
    info = {
        "num_samples": len(dataset),
        "input_shape": list(dataset.input_ids.shape),
        "attention_mask_shape": list(dataset.attention_mask.shape),
        "labels_shape": list(dataset.labels.shape),
        "teacher_logits_shape": list(dataset.teacher_logits.shape),
        "input_dtype": str(dataset.input_ids.dtype),
        "teacher_logits_dtype": str(dataset.teacher_logits.dtype),
    }

    with open(save_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Dataset info saved to: {save_path}")


def load_dataset_info(save_path: str) -> Dict[str, Any]:
    """
    Load dataset information from file.

    Args:
        save_path: Path to the info file

    Returns:
        Dictionary containing dataset information
    """
    with open(save_path, "r") as f:
        info = json.load(f)

    return info


def validate_dataset(dataset: StudentDataset) -> bool:
    """
    Validate dataset for consistency.

    Args:
        dataset: StudentDataset to validate

    Returns:
        True if dataset is valid, False otherwise
    """
    try:
        # Check shapes
        if not (
            dataset.input_ids.shape[0]
            == dataset.attention_mask.shape[0]
            == dataset.labels.shape[0]
            == dataset.teacher_logits.shape[0]
        ):
            logger.error("Batch dimensions don't match")
            return False

        # Check sequence lengths
        if not (
            dataset.input_ids.shape[1]
            == dataset.attention_mask.shape[1]
            == dataset.labels.shape[1]
        ):
            logger.error("Sequence lengths don't match")
            return False

        # Check teacher logits sequence length
        if dataset.teacher_logits.shape[1] != dataset.input_ids.shape[1]:
            logger.error(
                "Teacher logits sequence length doesn't match input sequence length"
            )
            return False

        # Check for NaN or Inf values
        if torch.isnan(dataset.input_ids).any():
            logger.error("NaN values found in input_ids")
            return False

        if torch.isnan(dataset.teacher_logits).any():
            logger.error("NaN values found in teacher_logits")
            return False

        if torch.isinf(dataset.teacher_logits).any():
            logger.error("Inf values found in teacher_logits")
            return False

        logger.info("Dataset validation passed")
        return True

    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False


def create_dataloader(
    dataset: Union[StudentDataset, torch.utils.data.Subset],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for the dataset.

    Args:
        dataset: Dataset to create DataLoader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        DataLoader object
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
            "teacher_logits": torch.stack([item["teacher_logits"] for item in x]),
        },
    )


# Example usage and testing functions
def test_data_utils():
    """Test the data utilities."""
    logger.info("Testing data utilities...")

    # Create dummy data
    batch_size, seq_len, vocab_size = 4, 10, 1000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test StudentDataset
    dataset = StudentDataset(input_ids, attention_mask, labels, teacher_logits)
    logger.info(f"Dataset length: {len(dataset)}")

    # Test dataset item access
    item = dataset[0]
    logger.info(f"Item keys: {item.keys()}")
    logger.info(f"Input shape: {item['input_ids'].shape}")

    # Test validation
    is_valid = validate_dataset(dataset)
    logger.info(f"Dataset validation: {is_valid}")

    # Test DataLoader
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Batch input shape: {batch['input_ids'].shape}")

    logger.info("All data utility tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_data_utils()
