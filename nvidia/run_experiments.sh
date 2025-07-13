#!/bin/bash

# Test pure euclidean distance loss
echo "Testing pure euclidean distance loss (no cosine similarity)..."

# Fixed parameters based on previous findings
BATCH_SIZE=4  # Optimal from previous experiments
BOTTLENECK_SIZE=128
VECTOR_LENGTH=512
EPOCHS=20 # Quick test run
LEARNING_RATE=0.00001

# Parameters no longer used but needed for compatibility
MAX_HOME_DISTANCE=.1  # UNUSED - kept for compatibility
POSITION_WEIGHT=10.0   # UNUSED - kept for compatibility

echo "Pure euclidean distance test parameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Bottleneck: $BOTTLENECK_SIZE"
echo "  Epochs: $EPOCHS (quick test)"
echo "  Learning rate: $LEARNING_RATE"
echo "  NOTE: max_home_distance and position_weight are now unused"
echo ""

python experiments/parameter_sweep.py \
    --bottleneck-size $BOTTLENECK_SIZE \
    --vector-length $VECTOR_LENGTH \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --max-home-distance $MAX_HOME_DISTANCE \
    --position-weight $POSITION_WEIGHT \
    --vector-method random

echo "Pure euclidean distance test completed!"
echo "Results saved in /models/ - should show perfect diversity!" 