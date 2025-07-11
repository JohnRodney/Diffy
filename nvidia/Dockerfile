# Use PyTorch base image that works with RTX 5090
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
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

# Verify CUDA and numba installation
RUN python -c "import numba; print('Numba version:', numba.__version__)" && \
    python -c "from numba import cuda; print('CUDA available:', cuda.is_available())" && \
    python -c "import numpy as np; print('NumPy version:', np.__version__)"

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Create results directory
RUN mkdir -p /app/results

# Add GPU detection script
RUN echo '#!/bin/bash\n\
echo "🔍 GPU Detection Report"\n\
echo "====================="\n\
nvidia-smi\n\
echo ""\n\
echo "🐍 Python CUDA Test"\n\
echo "=================="\n\
python -c "from numba import cuda; print(f\"CUDA available: {cuda.is_available()}\"); print(f\"CUDA devices: {cuda.list_devices()}\")" || echo "⚠️ CUDA test failed"\n\
echo ""\n\
echo "🚀 Starting parameter sweep..."\n\
python experiments/parameter_sweep.py\n\
' > /app/run_with_gpu_check.sh && chmod +x /app/run_with_gpu_check.sh

# Health check to verify RTX 5090 is detected
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from numba import cuda; assert cuda.is_available(), 'CUDA not available'" || exit 1

# Run with GPU detection
CMD ["/app/run_with_gpu_check.sh"] 