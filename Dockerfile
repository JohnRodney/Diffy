FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch nightly builds for latest GPU support
RUN pip install --no-cache-dir --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your codebase
COPY . .

# Create models directory for saving VAE weights and outputs
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Create VAE training startup script
RUN printf '#!/bin/bash\\necho \"Starting VAE training...\"\\npython train_vae.py\\necho \"VAE training completed!\"\\n' > /app/start_vae_training.sh && chmod +x /app/start_vae_training.sh

# Default command: run VAE training
CMD ["/app/start_vae_training.sh"]