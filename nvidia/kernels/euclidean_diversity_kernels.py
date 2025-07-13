#!/usr/bin/env python3
"""
Euclidean distance diversity penalty CUDA kernels
Prevents mode collapse by penalizing geometric clustering
Uses cosine similarity for reconstruction + Euclidean distance for diversity
"""

import numba
from numba import cuda
import math

# Euclidean diversity hyperparameters
EUCLIDEAN_DIVERSITY_WEIGHT = 0.1
MIN_DISTANCE_THRESHOLD = 0.1  # Minimum Euclidean distance before penalty

@cuda.jit
def gpu_euclidean_diversity_penalty(predicted, diversity_loss, min_distance_threshold):
    """
    Euclidean distance diversity penalty: penalize when outputs are too close geometrically
    
    predicted: (batch_size, vector_size)
    diversity_loss: (1,) - scalar output
    min_distance_threshold: minimum Euclidean distance before penalty kicks in
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    # Use one thread per sample pair (following existing pattern)
    pair_idx = cuda.grid(1)
    total_pairs = batch_size * (batch_size - 1) // 2
    
    if pair_idx < total_pairs:
        # Convert linear index to (i, j) pair where i < j
        i = 0
        j = 1
        remaining = pair_idx
        
        while remaining >= (batch_size - i - 1):
            remaining -= (batch_size - i - 1)
            i += 1
            j = i + 1
        
        j += remaining
        
        # Compute Euclidean distance between samples i and j
        squared_distance = 0.0
        
        for dim in range(vector_size):
            diff = predicted[i, dim] - predicted[j, dim]
            squared_distance += diff * diff
        
        # Avoid sqrt of very small numbers for numerical stability
        if squared_distance > 1e-16:
            euclidean_distance = math.sqrt(squared_distance)
            
            # Penalty only if distance is below threshold
            if euclidean_distance < min_distance_threshold:
                penalty = (min_distance_threshold - euclidean_distance) * (min_distance_threshold - euclidean_distance)
                cuda.atomic.add(diversity_loss, 0, penalty)

@cuda.jit
def gpu_euclidean_diversity_gradient(predicted, diversity_gradient, diversity_weight, min_distance_threshold):
    """
    Euclidean distance diversity gradient: gradient of penalty w.r.t. each output dimension
    
    predicted: (batch_size, vector_size)
    diversity_gradient: (batch_size, vector_size)
    diversity_weight: scaling factor for penalty
    min_distance_threshold: minimum distance threshold
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.blockIdx.x
    dim_idx = cuda.threadIdx.x
    
    if sample_idx < batch_size and dim_idx < vector_size:
        current_val = predicted[sample_idx, dim_idx]
        
        # Compute gradient by summing contributions from all other samples
        grad_sum = 0.0
        
        for other_idx in range(batch_size):
            if other_idx != sample_idx:
                # Compute Euclidean distance to other sample
                squared_distance = 0.0
                
                for dim in range(vector_size):
                    diff = predicted[sample_idx, dim] - predicted[other_idx, dim]
                    squared_distance += diff * diff
                
                # Numerical stability check
                if squared_distance > 1e-16:
                    euclidean_distance = math.sqrt(squared_distance)
                    
                    # Only apply gradient if distance is below threshold
                    if euclidean_distance < min_distance_threshold:
                        # Gradient of penalty w.r.t current dimension
                        # d/dx[(threshold - distance)²] = -2(threshold - distance) * (x_i - x_j)/distance
                        diff_current_dim = current_val - predicted[other_idx, dim_idx]
                        penalty_magnitude = min_distance_threshold - euclidean_distance
                        grad_contribution = -2.0 * penalty_magnitude * diff_current_dim / euclidean_distance
                        grad_sum += grad_contribution
        
        # Apply diversity weight and store gradient
        diversity_gradient[sample_idx, dim_idx] = diversity_weight * grad_sum

@cuda.jit
def gpu_combined_cosine_euclidean_loss(predicted, target, combined_loss, diversity_weight, min_distance_threshold):
    """
    Combined loss: cosine similarity reconstruction + Euclidean distance diversity penalty
    
    predicted: (batch_size, vector_size)
    target: (batch_size, vector_size)
    combined_loss: (batch_size,)
    diversity_weight: weight for diversity penalty
    min_distance_threshold: minimum Euclidean distance threshold
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.grid(1)
    
    if sample_idx < batch_size:
        # Compute reconstruction loss (cosine similarity)
        dot_product = 0.0
        norm_pred_sq = 0.0
        norm_target_sq = 0.0
        
        for dim in range(vector_size):
            pred_val = predicted[sample_idx, dim]
            target_val = target[sample_idx, dim]
            
            dot_product += pred_val * target_val
            norm_pred_sq += pred_val * pred_val
            norm_target_sq += target_val * target_val
        
        norm_pred = math.sqrt(norm_pred_sq)
        norm_target = math.sqrt(norm_target_sq)
        
        reconstruction_loss = 0.0
        if norm_pred > 1e-8 and norm_target > 1e-8:
            cos_sim = dot_product / (norm_pred * norm_target)
            reconstruction_loss = 1.0 - cos_sim
        else:
            reconstruction_loss = 1.0
        
        # Compute diversity penalty for this sample
        diversity_penalty = 0.0
        
        for other_idx in range(batch_size):
            if other_idx != sample_idx:
                # Compute Euclidean distance to other sample
                squared_distance = 0.0
                
                for dim in range(vector_size):
                    diff = predicted[sample_idx, dim] - predicted[other_idx, dim]
                    squared_distance += diff * diff
                
                if squared_distance > 1e-16:
                    euclidean_distance = math.sqrt(squared_distance)
                    
                    if euclidean_distance < min_distance_threshold:
                        penalty = (min_distance_threshold - euclidean_distance) * (min_distance_threshold - euclidean_distance)
                        diversity_penalty += penalty
        
        # Average the diversity penalty across all other samples
        if batch_size > 1:
            diversity_penalty /= (batch_size - 1)
        
        combined_loss[sample_idx] = reconstruction_loss + diversity_weight * diversity_penalty

def launch_euclidean_diversity_penalty(predicted, diversity_loss, min_distance_threshold=MIN_DISTANCE_THRESHOLD):
    """
    Launch Euclidean distance diversity penalty computation
    
    predicted: (batch_size, vector_size)
    diversity_loss: (1,)
    """
    batch_size = predicted.shape[0]
    total_pairs = batch_size * (batch_size - 1) // 2
    
    if total_pairs > 0:
        # Use grid pattern from documentation
        threadsperblock = 256
        blockspergrid = (total_pairs + threadsperblock - 1) // threadsperblock
        
        # Initialize diversity_loss to zero
        diversity_loss[0] = 0.0
        
        gpu_euclidean_diversity_penalty[blockspergrid, threadsperblock](
            predicted, diversity_loss, min_distance_threshold
        )

def launch_euclidean_diversity_gradient(predicted, diversity_gradient, diversity_weight=EUCLIDEAN_DIVERSITY_WEIGHT, min_distance_threshold=MIN_DISTANCE_THRESHOLD):
    """
    Launch Euclidean distance diversity gradient computation
    
    predicted: (batch_size, vector_size)
    diversity_gradient: (batch_size, vector_size)
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    # Use 2D grid: blocks for samples, threads for dimensions
    threadsperblock = min(256, vector_size)
    blockspergrid = batch_size
    
    gpu_euclidean_diversity_gradient[blockspergrid, threadsperblock](
        predicted, diversity_gradient, diversity_weight, min_distance_threshold
    )

def launch_combined_cosine_euclidean_loss(predicted, target, combined_loss, diversity_weight=EUCLIDEAN_DIVERSITY_WEIGHT, min_distance_threshold=MIN_DISTANCE_THRESHOLD):
    """
    Launch combined cosine similarity reconstruction + Euclidean distance diversity loss
    
    predicted: (batch_size, vector_size)
    target: (batch_size, vector_size)
    combined_loss: (batch_size,)
    """
    batch_size = predicted.shape[0]
    
    # Use 1D grid pattern from documentation
    threadsperblock = 256
    blockspergrid = (batch_size + threadsperblock - 1) // threadsperblock
    
    gpu_combined_cosine_euclidean_loss[blockspergrid, threadsperblock](
        predicted, target, combined_loss, diversity_weight, min_distance_threshold
    ) 