# Use PyTorch base image that works with RTX 5090
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies with RTX 5090 support
RUN pip install --no-cache-dir \
    numpy>=2.0.0 \
    numba>=0.59.0 \
    datasets>=4.0.0

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Copy nvidia directory with training code
COPY . .

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Create /models directory for all outputs (fixes root permission issue)
RUN mkdir -p /models

# Make experiment script executable
RUN chmod +x run_experiments.sh

# Health check to verify RTX 5090 is detected
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from numba import cuda; assert cuda.is_available(), 'CUDA not available'" || exit 1

CMD ["./run_experiments.sh"]
