#!/usr/bin/env python3
"""
Diversification penalty CUDA kernels for preventing mode collapse
Penalizes when model outputs are too similar to each other
"""

import numba
from numba import cuda
import math

# Diversification hyperparameters
DIVERSITY_WEIGHT = 0.1
SIMILARITY_THRESHOLD = 0.8

@cuda.jit
def gpu_batch_diversity_penalty(predicted, diversity_loss, similarity_threshold):
    """
    Batch diversity penalty: penalize high pairwise similarities
    
    predicted: (batch_size, vector_size)
    diversity_loss: (1,) - scalar output
    similarity_threshold: minimum similarity before penalty kicks in
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    # Use one thread per sample pair
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
        
        # Compute cosine similarity between samples i and j
        dot_product = 0.0
        norm_i_sq = 0.0
        norm_j_sq = 0.0
        
        for dim in range(vector_size):
            val_i = predicted[i, dim]
            val_j = predicted[j, dim]
            
            dot_product += val_i * val_j
            norm_i_sq += val_i * val_i
            norm_j_sq += val_j * val_j
        
        norm_i = math.sqrt(norm_i_sq)
        norm_j = math.sqrt(norm_j_sq)
        
        if norm_i > 1e-8 and norm_j > 1e-8:
            cos_sim = dot_product / (norm_i * norm_j)
            
            # Penalty only if similarity exceeds threshold
            if cos_sim > similarity_threshold:
                penalty = (cos_sim - similarity_threshold) * (cos_sim - similarity_threshold)
                cuda.atomic.add(diversity_loss, 0, penalty)

@cuda.jit
def gpu_batch_diversity_gradient(predicted, diversity_gradient, diversity_weight, similarity_threshold):
    """
    Batch diversity gradient: gradient of diversity penalty w.r.t. each output
    
    predicted: (batch_size, vector_size)
    diversity_gradient: (batch_size, vector_size)
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
                # Compute cosine similarity between current and other sample
                dot_product = 0.0
                norm_curr_sq = 0.0
                norm_other_sq = 0.0
                
                for dim in range(vector_size):
                    curr_val = predicted[sample_idx, dim]
                    other_val = predicted[other_idx, dim]
                    
                    dot_product += curr_val * other_val
                    norm_curr_sq += curr_val * curr_val
                    norm_other_sq += other_val * other_val
                
                norm_curr = math.sqrt(norm_curr_sq)
                norm_other = math.sqrt(norm_other_sq)
                
                if norm_curr > 1e-8 and norm_other > 1e-8:
                    cos_sim = dot_product / (norm_curr * norm_other)
                    
                    # Only contribute gradient if similarity exceeds threshold
                    if cos_sim > similarity_threshold:
                        # Gradient of (cos_sim - threshold)^2 w.r.t. current_val
                        other_val = predicted[other_idx, dim_idx]
                        norm_product = norm_curr * norm_other
                        
                        # d/dx[cos_sim] = (other - current * cos_sim) / (|current| * |other|)
                        cos_sim_grad = (other_val - current_val * cos_sim) / norm_product
                        
                        # d/dx[(cos_sim - threshold)^2] = 2 * (cos_sim - threshold) * cos_sim_grad
                        penalty_grad = 2.0 * (cos_sim - similarity_threshold) * cos_sim_grad
                        grad_sum += penalty_grad
        
        # Normalize by number of pairs and apply diversity weight
        if batch_size > 1:
            grad_sum /= (batch_size - 1)
        
        diversity_gradient[sample_idx, dim_idx] = diversity_weight * grad_sum

@cuda.jit
def gpu_combined_diversity_loss(predicted, target, combined_loss, diversity_weight, similarity_threshold):
    """
    Combined reconstruction + diversity loss
    
    predicted: (batch_size, vector_size)
    target: (batch_size, vector_size)
    combined_loss: (batch_size,)
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.blockIdx.x
    
    if sample_idx < batch_size:
        # Compute reconstruction loss (cosine similarity)
        dot_product = 0.0
        pred_norm_sq = 0.0
        target_norm_sq = 0.0
        
        for dim in range(vector_size):
            pred_val = predicted[sample_idx, dim]
            target_val = target[sample_idx, dim]
            
            dot_product += pred_val * target_val
            pred_norm_sq += pred_val * pred_val
            target_norm_sq += target_val * target_val
        
        pred_norm = math.sqrt(pred_norm_sq)
        target_norm = math.sqrt(target_norm_sq)
        
        reconstruction_loss = 0.0
        if pred_norm > 1e-8 and target_norm > 1e-8:
            cos_sim = dot_product / (pred_norm * target_norm)
            reconstruction_loss = 1.0 - cos_sim
        else:
            reconstruction_loss = 1.0
        
        # Compute diversity penalty for this sample
        diversity_penalty = 0.0
        penalty_count = 0
        
        for other_idx in range(batch_size):
            if other_idx != sample_idx:
                other_dot = 0.0
                other_norm_sq = 0.0
                
                for dim in range(vector_size):
                    pred_val = predicted[sample_idx, dim]
                    other_val = predicted[other_idx, dim]
                    
                    other_dot += pred_val * other_val
                    other_norm_sq += other_val * other_val
                
                other_norm = math.sqrt(other_norm_sq)
                
                if pred_norm > 1e-8 and other_norm > 1e-8:
                    other_cos_sim = other_dot / (pred_norm * other_norm)
                    
                    if other_cos_sim > similarity_threshold:
                        penalty = (other_cos_sim - similarity_threshold) * (other_cos_sim - similarity_threshold)
                        diversity_penalty += penalty
                        penalty_count += 1
        
        if penalty_count > 0:
            diversity_penalty /= penalty_count
        
        combined_loss[sample_idx] = reconstruction_loss + diversity_weight * diversity_penalty

def launch_batch_diversity_penalty(predicted, diversity_loss, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Launch batch diversity penalty computation
    """
    batch_size = predicted.shape[0]
    total_pairs = batch_size * (batch_size - 1) // 2
    
    threads_per_block = 256
    blocks_per_grid = math.ceil(total_pairs / threads_per_block)
    
    # Initialize diversity_loss to zero
    diversity_loss[0] = 0.0
    
    gpu_batch_diversity_penalty[blocks_per_grid, threads_per_block](
        predicted, diversity_loss, similarity_threshold
    )
    cuda.synchronize()

def launch_batch_diversity_gradient(predicted, diversity_gradient, diversity_weight=DIVERSITY_WEIGHT, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Launch batch diversity gradient computation
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    threads_per_block = min(vector_size, 512)
    blocks_per_grid = batch_size
    
    gpu_batch_diversity_gradient[blocks_per_grid, threads_per_block](
        predicted, diversity_gradient, diversity_weight, similarity_threshold
    )
    cuda.synchronize()

def launch_combined_diversity_loss(predicted, target, combined_loss, diversity_weight=DIVERSITY_WEIGHT, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Launch combined reconstruction + diversity loss computation
    """
    batch_size = predicted.shape[0]
    
    threads_per_block = 1
    blocks_per_grid = batch_size
    
    gpu_combined_diversity_loss[blocks_per_grid, threads_per_block](
        predicted, target, combined_loss, diversity_weight, similarity_threshold
    )
    cuda.synchronize() 