# SteveAI - Model Distillation Project

## Overview

SteveAI is a project focused on performing model distillation, initially targeting the compression of large language models. This repository currently contains the first stage: **Teacher Model Inference**.

The goal is to make model distillation feasible even with limited computational resources (like free tiers on Kaggle or Colab) by breaking the process into stages.

*   **Teacher Model:** `deepseek-ai/deepseek-coder-6.7b-instruct`
*   **Student Model (Target):** `deepseek-ai/deepseek-coder-1.3b-instruct`

**Current Stage:** This code performs inference using the large teacher model (`6.7B`) on a dataset and saves the resulting logits. This is computationally intensive and is designed to run efficiently on a GPU environment like Kaggle.

**Motivation:** Running inference for large models requires significant RAM and computational power. Free CPU environments often lack sufficient RAM. This script leverages free GPU resources (like Kaggle Kernels) to generate the necessary teacher outputs (logits), which can then be used in a less resource-intensive environment to train the smaller student model.

## Prerequisites

*   A Kaggle account (for using free GPU resources).
*   Basic understanding of Python, PyTorch, and Transformers.
*   Familiarity with Kaggle Notebooks.

## Setup and Execution (Kaggle GPU)

1.  **Clone or Download:** Get the project files onto your local machine or directly upload them to Kaggle.
    ```bash
    git clone https://github.com/your-username/SteveAI.git
    cd SteveAI
    ```
    (Replace `your-username` with your actual GitHub username after creating the repository)

2.  **Kaggle Notebook Setup:**
    *   Create a new Kaggle Notebook.
    *   **Upload:** Upload the `teacher_inference_gpu.py` script and the `requirements.txt` file to your notebook environment.
    *   **Accelerator:** In the Notebook settings (right panel), select a **GPU accelerator** (e.g., T4 x2 or P100).
    *   **Internet:** Ensure the **Internet** toggle is set to **ON**.

3.  **Install Dependencies:** In a code cell in your Kaggle Notebook, run:
    ```python
    !pip install -r requirements.txt -q
    ```

4.  **Run the Inference Script:** In another code cell, execute the main script:
    ```python
    !python teacher_inference_gpu.py
    ```
    *(Alternatively, you can paste the content of `teacher_inference_gpu.py` directly into a notebook cell and run it.)*

5.  **Configuration:** You can modify the parameters in the `Config` class at the top of `teacher_inference_gpu.py` or use the `config.yaml` file:
    *   `TEACHER_MODEL_ID`: The Hugging Face ID of the teacher model.
    *   `TOKENIZER_PATH`: Path/ID for the tokenizer (usually same as teacher).
    *   `DATASET_ID`: Hugging Face dataset ID (e.g., `yahma/alpaca-cleaned`).
    *   `DATASET_SUBSET_SIZE`: Number of samples to process (e.g., `500`). Set to `None` for the full dataset (be mindful of time limits).
    *   `MAX_SEQ_LENGTH`: Max token sequence length (e.g., `256`). Higher values need more VRAM.
    *   `TEACHER_INFERENCE_BATCH_SIZE`: Number of samples per batch (e.g., `4`). Adjust based on GPU VRAM (T4 might need 2 or 4, V100+ could handle 8+).
    *   Output directory is automatically detected based on environment (Kaggle vs local).

6.  **Output:** The script will generate and save teacher logits as `.pt` files in the specified output directory (e.g., `/kaggle/working/Kaggle_Distillation_GPU_Teacher_Inference/teacher_logits_float16/`). Progress and memory usage will be logged.

7.  **Save Results:** Once the script finishes:
    *   Click "Save Version" in the top-right of the Kaggle editor.
    *   Select "Save & Run All (Commit)".
    *   Wait for the commit to complete. The output files will be saved as a Kaggle Dataset linked to this notebook version. You can then add this output dataset as input to a *new* notebook for the student training phase.

## Next Steps

*   Develop the second script (`student_training.py` - TBD) which will load the pre-computed teacher logits and train the student model. This stage might be runnable on CPU or GPU depending on the student model size and training configuration.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.