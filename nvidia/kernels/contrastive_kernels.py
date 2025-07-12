#!/usr/bin/env python3
"""
Contrastive loss CUDA kernels for preventing mode collapse
Maximizes similarity to target while minimizing similarity to non-targets
"""

import numba
from numba import cuda
import math

# Contrastive loss hyperparameters
MARGIN = 0.2
NEGATIVE_WEIGHT = 0.5

@cuda.jit
def gpu_contrastive_loss(predicted, target, batch_indices, loss_output, margin, neg_weight):
    """
    Contrastive loss: (1 - cos_sim(pred, target)) + λ * max(0, cos_sim(pred, negatives) - margin)
    
    predicted: (batch_size, vector_size)
    target: (batch_size, vector_size) 
    batch_indices: (batch_size,) - indices for negative sampling
    loss_output: (batch_size,)
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.blockIdx.x
    
    if sample_idx < batch_size:
        # Compute positive term: 1 - cos_sim(pred, target)
        pos_dot = 0.0
        pred_norm_sq = 0.0
        target_norm_sq = 0.0
        
        for dim in range(vector_size):
            pred_val = predicted[sample_idx, dim]
            target_val = target[sample_idx, dim]
            
            pos_dot += pred_val * target_val
            pred_norm_sq += pred_val * pred_val
            target_norm_sq += target_val * target_val
        
        pred_norm = math.sqrt(pred_norm_sq)
        target_norm = math.sqrt(target_norm_sq)
        
        positive_loss = 0.0
        if pred_norm > 1e-8 and target_norm > 1e-8:
            pos_cos_sim = pos_dot / (pred_norm * target_norm)
            positive_loss = 1.0 - pos_cos_sim
        else:
            positive_loss = 1.0
        
        # Compute negative term: max(0, cos_sim(pred, negatives) - margin)
        negative_loss = 0.0
        negative_count = 0
        
        for neg_idx in range(batch_size):
            if neg_idx != sample_idx:  # Don't compare with self
                neg_dot = 0.0
                neg_norm_sq = 0.0
                
                for dim in range(vector_size):
                    pred_val = predicted[sample_idx, dim]
                    neg_val = target[neg_idx, dim]  # Use other targets as negatives
                    
                    neg_dot += pred_val * neg_val
                    neg_norm_sq += neg_val * neg_val
                
                neg_norm = math.sqrt(neg_norm_sq)
                
                if pred_norm > 1e-8 and neg_norm > 1e-8:
                    neg_cos_sim = neg_dot / (pred_norm * neg_norm)
                    negative_loss += max(0.0, neg_cos_sim - margin)
                    negative_count += 1
        
        if negative_count > 0:
            negative_loss /= negative_count
        
        loss_output[sample_idx] = positive_loss + neg_weight * negative_loss

@cuda.jit
def gpu_contrastive_gradient(predicted, target, batch_indices, gradient, margin, neg_weight):
    """
    Contrastive gradient computation
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.blockIdx.x
    dim_idx = cuda.threadIdx.x
    
    if sample_idx < batch_size and dim_idx < vector_size:
        pred_val = predicted[sample_idx, dim_idx]
        target_val = target[sample_idx, dim_idx]
        
        # Compute norms for current sample
        pred_norm_sq = 0.0
        target_norm_sq = 0.0
        pos_dot = 0.0
        
        for dim in range(vector_size):
            p_val = predicted[sample_idx, dim]
            t_val = target[sample_idx, dim]
            pred_norm_sq += p_val * p_val
            target_norm_sq += t_val * t_val
            pos_dot += p_val * t_val
        
        pred_norm = math.sqrt(pred_norm_sq)
        target_norm = math.sqrt(target_norm_sq)
        
        # Positive gradient: -(target - predicted * cos_sim) / (|pred| * |target|)
        pos_grad = 0.0
        if pred_norm > 1e-8 and target_norm > 1e-8:
            pos_cos_sim = pos_dot / (pred_norm * target_norm)
            norm_product = pred_norm * target_norm
            pos_grad = -(target_val - pred_val * pos_cos_sim) / norm_product
        
        # Negative gradient: sum over all negatives
        neg_grad = 0.0
        negative_count = 0
        
        for neg_idx in range(batch_size):
            if neg_idx != sample_idx:
                neg_dot = 0.0
                neg_norm_sq = 0.0
                
                for dim in range(vector_size):
                    p_val = predicted[sample_idx, dim]
                    n_val = target[neg_idx, dim]
                    neg_dot += p_val * n_val
                    neg_norm_sq += n_val * n_val
                
                neg_norm = math.sqrt(neg_norm_sq)
                
                if pred_norm > 1e-8 and neg_norm > 1e-8:
                    neg_cos_sim = neg_dot / (pred_norm * neg_norm)
                    
                    if neg_cos_sim > margin:  # Only contribute if violating margin
                        neg_val = target[neg_idx, dim_idx]
                        norm_product = pred_norm * neg_norm
                        neg_grad += (neg_val - pred_val * neg_cos_sim) / norm_product
                        negative_count += 1
        
        if negative_count > 0:
            neg_grad /= negative_count
        
        gradient[sample_idx, dim_idx] = pos_grad + neg_weight * neg_grad

def launch_contrastive_loss(predicted, target, batch_indices, loss_output, margin=MARGIN, neg_weight=NEGATIVE_WEIGHT):
    """
    Launch contrastive loss computation
    """
    batch_size = predicted.shape[0]
    
    threads_per_block = 1
    blocks_per_grid = batch_size
    
    gpu_contrastive_loss[blocks_per_grid, threads_per_block](
        predicted, target, batch_indices, loss_output, margin, neg_weight
    )
    cuda.synchronize()

def launch_contrastive_gradient(predicted, target, batch_indices, gradient, margin=MARGIN, neg_weight=NEGATIVE_WEIGHT):
    """
    Launch contrastive gradient computation
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    threads_per_block = min(vector_size, 512)
    blocks_per_grid = batch_size
    
    gpu_contrastive_gradient[blocks_per_grid, threads_per_block](
        predicted, target, batch_indices, gradient, margin, neg_weight
    )
    cuda.synchronize() 