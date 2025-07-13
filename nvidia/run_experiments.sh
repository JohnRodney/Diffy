#!/bin/bash

# Test pure euclidean distance loss with PCA initialization on tokenized color data
echo "Testing pure euclidean distance loss + PCA weight initialization on tokenized color data..."

# Fixed parameters based on previous findings
BATCH_SIZE=4  # Optimal from previous experiments
BOTTLENECK_SIZE=128
VECTOR_LENGTH=512
EPOCHS=100000
LEARNING_RATE=0.001

echo "Simple MSE + embedding/PCA initialization test parameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Bottleneck: $BOTTLENECK_SIZE"
echo "  Epochs: $EPOCHS (quick test)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Data: Tokenized color names with learnable embeddings"
echo "  Initialization: Xavier for embeddings, PCA for autoencoder weights"
echo "  Loss: Simple MSE reconstruction"
echo ""

    python experiments/parameter_sweep.py \
        --bottleneck-size $BOTTLENECK_SIZE \
        --vector-length $VECTOR_LENGTH \
        --epochs $EPOCHS \
        --learning-rate $LEARNING_RATE \
        --batch-size $BATCH_SIZE \
        --vector-method tokenized
    
echo "Simple MSE + embeddings test completed!"
echo "Results saved in /models/ - should show clean learning without numerical issues!" 