#!/bin/bash
# Launch script for GPU parameter sweep on RTX 5090 server

set -e

echo "🚀 Launching GPU Parameter Sweep on RTX 5090"
echo "================================================"

# Configuration
CONTAINER_NAME="diffy-gpu-sweep"
RESULTS_DIR="$(pwd)/gpu_sweep_results"
RUNTIME_HOURS=20

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR"
echo "⏰ Runtime: $RUNTIME_HOURS hours"

# Build Docker image
echo "🏗️  Building Docker image..."
docker build -t diffy-gpu:latest -f nvidia/docker/Dockerfile .

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Test RTX 5090 compatibility first
echo "🧪 Testing RTX 5090 compatibility..."
docker run --rm \
    --gpus all \
    --runtime nvidia \
    -e CUDA_VISIBLE_DEVICES=0 \
    diffy-gpu:latest \
    python3 nvidia/docker/test_rtx5090.py

if [ $? -ne 0 ]; then
    echo "❌ RTX 5090 compatibility test failed!"
    echo "🔧 Please check your NVIDIA drivers and CUDA installation"
    exit 1
fi

echo "✅ RTX 5090 compatibility confirmed!"

# Run parameter sweep
echo "🔥 Starting parameter sweep..."
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --runtime nvidia \
    -v "$RESULTS_DIR:/app/results" \
    -e MAX_RUNTIME_HOURS=$RUNTIME_HOURS \
    -e CUDA_VISIBLE_DEVICES=0 \
    diffy-gpu:latest

echo "✅ Parameter sweep started!"
echo "📊 Monitor progress:"
echo "   docker logs -f $CONTAINER_NAME"
echo ""
echo "🔍 Check results:"
echo "   ls -la $RESULTS_DIR"
echo ""
echo "⏹️  To stop early:"
echo "   docker stop $CONTAINER_NAME"
echo ""

# Show initial logs
echo "📋 Initial logs:"
echo "=================="
sleep 2
docker logs $CONTAINER_NAME

echo ""
echo "🎯 Parameter sweep is now running in the background!"
echo "   It will automatically stop after $RUNTIME_HOURS hours"
echo "   Results will be saved to: $RESULTS_DIR" 