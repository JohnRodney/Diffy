#!/bin/bash

# Test pure euclidean distance loss with PCA initialization on tokenized color data
echo "Testing pure euclidean distance loss + PCA weight initialization on tokenized color data..."

# Fixed parameters based on previous findings
BATCH_SIZE=4  # Optimal from previous experiments
BOTTLENECK_SIZE=128
VECTOR_LENGTH=512
EPOCHS=100000
LEARNING_RATE=0.001

# Parameters no longer used but needed for compatibility
MAX_HOME_DISTANCE=.1  # UNUSED - kept for compatibility
POSITION_WEIGHT=10.0   # UNUSED - kept for compatibility

echo "Pure euclidean distance + PCA initialization test parameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Bottleneck: $BOTTLENECK_SIZE"
echo "  Epochs: $EPOCHS (quick test)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Data: Tokenized color names (not random vectors)"
echo "  Initialization: PCA-based encoder/decoder weights"
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
        --vector-method tokenized
    
echo "Pure euclidean distance + PCA initialization test completed!"
echo "Results saved in /models/ - should show near-perfect diversity from epoch 0!" 