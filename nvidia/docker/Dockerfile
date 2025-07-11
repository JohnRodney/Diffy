FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install latest numba with LATEST numba-cuda package for RTX 5090 Blackwell support
RUN pip install --no-cache-dir --upgrade numba numba-cuda

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the GPU training code
COPY . .

# Set environment variables for RTX 5090 support
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV NUMBA_CUDA_DEBUGINFO=1
ENV NUMBA_ENABLE_CUDASIM=0

# Create volume directories
RUN mkdir -p /app/models /app/logs /app/generated_images

# Default command
CMD ["python", "experiments/parameter_sweep.py"] 