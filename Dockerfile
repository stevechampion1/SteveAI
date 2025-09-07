# SteveAI - Dockerfile

# Use official PyTorch image as base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies (optional)
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/output \
    /app/logs \
    /app/visualizations \
    /app/checkpoints

# Set permissions
RUN chmod +x *.py

# Create non-root user
RUN useradd -m -u 1000 steveai && \
    chown -R steveai:steveai /app
USER steveai

# Expose ports
EXPOSE 8000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Default command
CMD ["python", "teacher_inference_gpu.py"]

# Labels
LABEL maintainer="SteveAI Team <steveai@example.com>"
LABEL description="SteveAI - Knowledge Distillation Framework"
LABEL version="1.0.0"
