#!/usr/bin/env python3
"""
Home position penalty CUDA kernels
Prevents mode collapse by anchoring each vector to its "home" position
Uses cosine similarity for reconstruction + Euclidean distance to maintain position
"""

import numba
from numba import cuda
import math

# Home position hyperparameters
HOME_POSITION_WEIGHT = 1.0
MAX_HOME_DISTANCE = 2.0  # Maximum allowed distance from home before penalty

@cuda.jit
def gpu_home_position_penalty(predicted, training, position_loss, max_home_distance):
    """
    Home position penalty: penalize when outputs drift too far from their training position
    
    predicted: (batch_size, vector_size) - model outputs
    training: (batch_size, vector_size) - training vectors (home positions)
    position_loss: (1,) - scalar output
    max_home_distance: maximum allowed distance from home before penalty kicks in
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    # One thread per sample (much simpler than pairwise!)
    sample_idx = cuda.grid(1)
    
    if sample_idx < batch_size:
        # Compute Euclidean distance from predicted to home position
        squared_distance = 0.0
        
        for dim in range(vector_size):
            diff = predicted[sample_idx, dim] - training[sample_idx, dim]
            squared_distance += diff * diff
        
        distance = math.sqrt(squared_distance)
        
        # Apply penalty if too far from home
        if distance > max_home_distance:
            excess_distance = distance - max_home_distance
            penalty = excess_distance * excess_distance  # Quadratic penalty
            cuda.atomic.add(position_loss, 0, penalty)

@cuda.jit
def gpu_home_position_gradient(predicted, training, position_gradient, position_weight, max_home_distance):
    """
    Home position gradient: compute gradients for vectors that are too far from home
    
    predicted: (batch_size, vector_size) - model outputs
    training: (batch_size, vector_size) - training vectors (home positions)
    position_gradient: (batch_size, vector_size) - gradient output
    position_weight: strength of home position penalty
    max_home_distance: maximum allowed distance from home
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    # One thread per vector element
    sample_idx = cuda.blockIdx.x
    dim_idx = cuda.threadIdx.x
    
    if sample_idx < batch_size and dim_idx < vector_size:
        # Compute distance from home
        squared_distance = 0.0
        for d in range(vector_size):
            diff = predicted[sample_idx, d] - training[sample_idx, d]
            squared_distance += diff * diff
        
        distance = math.sqrt(squared_distance)
        
        # Only apply gradient if too far from home
        if distance > max_home_distance:
            # Gradient points back toward home
            diff = predicted[sample_idx, dim_idx] - training[sample_idx, dim_idx]
            
            if distance > 0:
                # Normalize and scale by penalty strength
                excess_distance = distance - max_home_distance
                gradient = position_weight * 2.0 * excess_distance * (diff / distance)
                position_gradient[sample_idx, dim_idx] = gradient
            else:
                position_gradient[sample_idx, dim_idx] = 0.0
        else:
            position_gradient[sample_idx, dim_idx] = 0.0

@cuda.jit
def gpu_combined_cosine_home_loss(predicted, target, training, combined_loss, position_weight, max_home_distance):
    """
    Euclidean distance loss: distance between predicted and training vectors
    
    predicted: (batch_size, vector_size) - model outputs
    target: (batch_size, vector_size) - target vectors (unused now)
    training: (batch_size, vector_size) - training vectors (home positions)
    combined_loss: (batch_size,) - per-sample output
    position_weight: unused now
    max_home_distance: unused now
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.grid(1)
    
    if sample_idx < batch_size:
        # Euclidean distance loss (primary loss)
        squared_distance = 0.0
        for dim in range(vector_size):
            diff = predicted[sample_idx, dim] - training[sample_idx, dim]
            squared_distance += diff * diff
        
        euclidean_loss = math.sqrt(squared_distance)
        
        combined_loss[sample_idx] = euclidean_loss

# Launch functions
def launch_home_position_penalty(predicted, training, position_loss, max_home_distance=MAX_HOME_DISTANCE):
    batch_size = predicted.shape[0]
    threads_per_block = 256
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    gpu_home_position_penalty[blocks_per_grid, threads_per_block](
        predicted, training, position_loss, max_home_distance
    )

def launch_home_position_gradient(predicted, training, position_gradient, position_weight=HOME_POSITION_WEIGHT, max_home_distance=MAX_HOME_DISTANCE):
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    gpu_home_position_gradient[batch_size, vector_size](
        predicted, training, position_gradient, position_weight, max_home_distance
    )

def launch_combined_cosine_home_loss(predicted, target, training, combined_loss, position_weight=HOME_POSITION_WEIGHT, max_home_distance=MAX_HOME_DISTANCE):
    batch_size = predicted.shape[0]
    threads_per_block = 256
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    gpu_combined_cosine_home_loss[blocks_per_grid, threads_per_block](
        predicted, target, training, combined_loss, position_weight, max_home_distance
    ) 