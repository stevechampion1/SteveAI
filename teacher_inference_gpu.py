# SteveAI - Teacher Model Inference Script (GPU Optimized)

# --- 1. 安装和导入 ---
# Dependencies are expected to be installed via requirements.txt
# Example Kaggle command: !pip install -r requirements.txt -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from tqdm.auto import tqdm
import gc, psutil, os, numpy as np, logging, time
import pandas as pd
import shutil
# import pyarrow is implicitly used by datasets

# --- Configuration ---
# Model Configuration
teacher_model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer_path = teacher_model_id # Usually same as teacher for inference consistency

# Dataset Configuration
dataset_id = "yahma/alpaca-cleaned" # Or other dataset like "tatsu-lab/alpaca"
dataset_subset_size = 500  # Number of samples (set to None for full dataset)
max_seq_length = 256      # Adjust based on VRAM and task needs (e.g., 128, 256, 512)

# GPU Inference Configuration
# Adjust based on available VRAM: T4/P100: 2 or 4; V100/A100: 8+
teacher_inference_batch_size = 4
dataloader_num_workers = 2 # Use multiple CPU cores for data loading

# Output Path Configuration (Kaggle specific - adjust if running elsewhere)
output_base_dir = "/kaggle/working/SteveAI_Teacher_Inference" # Project-named output
teacher_logits_path = os.path.join(output_base_dir, "teacher_logits_float16")
# --- End Configuration ---

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Function Definitions ---

def print_memory_usage(step_name=""):
    """Logs CPU RAM and GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    ram_used_gb = process.memory_info().rss / 1024**3
    ram_total_gb = psutil.virtual_memory().total / 1024**3
    ram_percent = psutil.virtual_memory().percent
    logger.info(f"[{step_name}] RAM Usage: {ram_percent:.1f}% ({ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB)")
    if torch.cuda.is_available():
        try:
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            # Get total memory from the primary device used by accelerate's 'auto' map if possible
            # Fallback to device 0 if direct query isn't straightforward
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            gpu_mem_total = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
            logger.info(f"[{step_name}] GPU-{device_index} VRAM: Allocated={gpu_mem_alloc:.2f} GB, Reserved={gpu_mem_reserved:.2f} GB, Total={gpu_mem_total:.2f} GB")
        except Exception as e:
            logger.warning(f"Could not get detailed GPU memory info: {e}")

def check_disk_space(path='/kaggle/working/'):
    """Logs free disk space in the specified path."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_space_gb = free / 1024**3
        logger.info(f"Free disk space in '{path}': {free_space_gb:.2f} GB")
        return free_space_gb
    except FileNotFoundError:
        logger.warning(f"Path not found for disk usage check: {path}")
        return 0

def create_prompt(example):
    """Creates a prompt string from an Alpaca-style dataset example."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "") # Include target output for full sequence logits

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

# --- Main Execution ---

def main():
    logger.info("--- Starting SteveAI Teacher Inference Script ---")
    os.makedirs(teacher_logits_path, exist_ok=True)
    logger.info(f"Teacher logits will be saved to: {teacher_logits_path}")

    # 4. Device Setup
    if not torch.cuda.is_available():
        logger.error("CUDA is NOT available. This script is designed for GPU execution.")
        logger.error("Running on CPU will be extremely slow and likely fail due to memory limits.")
        # Optionally exit if GPU is strictly required
        # return
        device = torch.device("cpu") # Allow CPU for testing, but warn heavily
    else:
        device = torch.device("cuda") # Primary device, 'auto' map handles distribution
        logger.info(f"CUDA is available. Using GPU(s). Primary device: {torch.cuda.get_device_name(device)}")

    print_memory_usage("Initial")
    check_disk_space()

    # 6. Load Tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    try:
        # trust_remote_code=True is necessary for deepseek-coder
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            logger.info("Tokenizer lacks pad_token, setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        # Padding side matters for causal LMs. 'right' is common for training.
        # Left padding is often preferred for generation, but for getting logits
        # of the full sequence matching training data, right padding is okay.
        tokenizer.padding_side = 'right'
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}", exc_info=True)
        logger.error("Ensure internet is ON in Kaggle settings and tokenizer path is correct.")
        raise

    # 7. Load and Preprocess Dataset
    logger.info(f"Loading dataset: {dataset_id} (Subset size: {dataset_subset_size})")
    try:
        split_string = "train"
        slice_str = f"[:{dataset_subset_size}]" if dataset_subset_size is not None else ""
        raw_dataset = load_dataset(dataset_id, split=f"{split_string}{slice_str}")
        logger.info(f"Loaded dataset with {len(raw_dataset)} examples.")

        prompt_dataset = raw_dataset.map(
            lambda example: {"prompt_text": create_prompt(example)},
            num_proc=dataloader_num_workers, # Use multiple cores if available
            desc="Creating prompts"
        )

        tokenized_dataset = prompt_dataset.map(
            lambda batch: tokenizer(
                batch["prompt_text"],
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors="pt" # Return PyTorch tensors
            ),
            batched=True,
            remove_columns=prompt_dataset.column_names,
            desc="Running tokenizer on dataset",
            num_proc=dataloader_num_workers
        )

        logger.info("Dataset tokenized and prepared for inference.")
        logger.info(f"Example tokenized keys: {tokenized_dataset.column_names}")
        logger.info(f"First tokenized example input_ids shape: {tokenized_dataset[0]['input_ids'].shape}")

    except Exception as e:
        logger.error(f"Error loading or processing dataset: {e}", exc_info=True)
        raise

    # 8. Teacher Model Inference Phase
    logger.info("--- Starting Teacher Inference Phase (GPU) ---")
    print_memory_usage("Before Teacher Load")
    teacher_model = None
    teacher_logits_files = []
    total_examples = len(tokenized_dataset)

    try:
        # Check if logits already exist
        logit_files_exist = True
        if not os.path.exists(teacher_logits_path) or not os.listdir(teacher_logits_path):
            logit_files_exist = False
        else:
            expected_num_batches = (total_examples + teacher_inference_batch_size - 1) // teacher_inference_batch_size
            existing_files = [f for f in os.listdir(teacher_logits_path) if f.startswith("teacher_logits_batch_") and f.endswith(".pt")]
            if len(existing_files) != expected_num_batches:
                logger.warning(f"Expected {expected_num_batches} logit batches, found {len(existing_files)}. Re-generating.")
                logit_files_exist = False
                shutil.rmtree(teacher_logits_path)
                os.makedirs(teacher_logits_path, exist_ok=True)
            else:
                 logger.info(f"Found {len(existing_files)} existing logit batch files. Assuming complete.")
                 teacher_logits_files = [os.path.join(teacher_logits_path, f"teacher_logits_batch_{i}.pt") for i in range(expected_num_batches)]

        if not logit_files_exist:
            logger.info(f"Loading Teacher Model: {teacher_model_id}")
            # Use device_map="auto" for efficient GPU usage (requires accelerate)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id,
                device_map="auto",
                torch_dtype=torch.float16, # Use float16 for less VRAM usage
                low_cpu_mem_usage=True,   # Helps reduce RAM spike during loading
                trust_remote_code=True
            )
            teacher_model.eval() # Set to evaluation mode
            logger.info("Teacher model loaded successfully onto GPU(s).")
            print_memory_usage("After Teacher Load")

            # Create DataLoader
            teacher_dataloader = DataLoader(
                tokenized_dataset,
                batch_size=teacher_inference_batch_size,
                collate_fn=default_data_collator, # Handles padding
                shuffle=False,
                num_workers=dataloader_num_workers,
                pin_memory=torch.cuda.is_available() # Pin memory if using GPU
            )

            logger.info(f"Generating teacher logits with batch size {teacher_inference_batch_size}...")
            inference_start_time = time.time()
            num_batches = len(teacher_dataloader)
            generated_files_count = 0

            for i, batch in enumerate(tqdm(teacher_dataloader, desc="Teacher Inference (GPU)", total=num_batches)):
                batch_start_time = time.time()
                # With device_map="auto", the model handles input placement.
                # Inputs should generally be on CPU before passing to model.
                # DataLoader with default_collate usually keeps them on CPU.
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                with torch.no_grad():
                    try:
                        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                        # Logits might be on GPU, move to CPU & convert for saving
                        teacher_logits = teacher_outputs.logits.to(device='cpu', dtype=torch.float16)
                    except Exception as e:
                        logger.error(f"Error during teacher inference on batch {i}: {e}", exc_info=True)
                        print_memory_usage(f"Teacher Inference Error Batch {i}")
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                             logger.error("CUDA Out of Memory! Try reducing teacher_inference_batch_size.")
                             raise e # Stop the process
                        logger.warning(f"Skipping batch {i} due to error.")
                        continue # Skip saving this batch

                # Save logits for the current batch
                save_path = os.path.join(teacher_logits_path, f"teacher_logits_batch_{i}.pt")
                torch.save({'teacher_logits': teacher_logits}, save_path)
                teacher_logits_files.append(save_path)
                generated_files_count += 1

                # Cleanup and Logging
                del input_ids, attention_mask, teacher_outputs, teacher_logits, batch
                if (i + 1) % 20 == 0 or i == num_batches - 1: # Log every 20 batches and at the end
                     gc.collect()
                     if torch.cuda.is_available():
                         torch.cuda.empty_cache()
                     print_memory_usage(f"Teacher Inference Batch {i+1}/{num_batches}")
                     batch_time_taken = time.time() - batch_start_time
                     logger.info(f"Batch {i} time: {batch_time_taken:.2f}s")
                     check_disk_space()
                     # Estimate remaining time
                     if i > 0:
                         elapsed_time = time.time() - inference_start_time
                         avg_time_per_batch = elapsed_time / (i + 1)
                         remaining_batches = num_batches - (i + 1)
                         estimated_remaining_time_min = (avg_time_per_batch * remaining_batches) / 60
                         logger.info(f"Estimated time remaining: {estimated_remaining_time_min:.2f} minutes")

            inference_end_time = time.time()
            total_minutes = (inference_end_time - inference_start_time) / 60
            logger.info(f"Teacher inference complete. Saved {generated_files_count} logit batch files.")
            logger.info(f"Total teacher inference time: {total_minutes:.2f} minutes")

        else:
            logger.info("Teacher logits already exist. Skipping inference.")
            # Ensure teacher_logits_files is populated correctly if skipping
            expected_num_batches = (total_examples + teacher_inference_batch_size - 1) // teacher_inference_batch_size
            teacher_logits_files = [os.path.join(teacher_logits_path, f"teacher_logits_batch_{i}.pt") for i in range(expected_num_batches)]


    except Exception as e:
        logger.error(f"An unhandled error occurred during the teacher inference phase: {e}", exc_info=True)
        print_memory_usage("Error During Teacher Inference")
        # Optional: re-raise to ensure the script stops on critical errors
        # raise e
    finally:
        # 9. Cleanup
        if 'teacher_model' in locals() and teacher_model is not None:
            logger.info("Unloading Teacher Model from GPU/RAM...")
            del teacher_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Teacher Model unloaded.")
            print_memory_usage("After Teacher Unload")
        else:
            logger.info("Teacher model was not loaded or already unloaded.")

    logger.info("--- Teacher Inference Script Finished ---")
    logger.info(f"Logits saved in: {teacher_logits_path}")
    if teacher_logits_files:
        logger.info(f"Total logit batches processed/found: {len(teacher_logits_files)}")
    check_disk_space()

# Entry point for the script
if __name__ == "__main__":
    main()