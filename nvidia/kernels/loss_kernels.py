#!/usr/bin/env python3
"""
MSE loss kernels for autoencoder training
Standard approach: MSE between input and output
"""

import numba
from numba import cuda
import math

@cuda.jit
def gpu_mse_loss(predicted, target, loss_output):
    """
    Simple MSE loss: (predicted - target)^2
    
    predicted: (batch_size, vector_size) - model outputs
    target: (batch_size, vector_size) - target vectors (inputs for autoencoder)
    loss_output: (batch_size,) - per-sample loss
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.grid(1)
    
    if sample_idx < batch_size:
        mse_loss = 0.0
        for dim in range(vector_size):
            diff = predicted[sample_idx, dim] - target[sample_idx, dim]
            mse_loss += diff * diff
        
        # Average over dimensions
        loss_output[sample_idx] = mse_loss / vector_size

@cuda.jit
def gpu_mse_gradient(predicted, target, gradient):
    """
    MSE gradient: 2 * (predicted - target) / vector_size
    
    predicted: (batch_size, vector_size) - model outputs
    target: (batch_size, vector_size) - target vectors
    gradient: (batch_size, vector_size) - gradient output
    """
    idx = cuda.grid(1)
    
    if idx < predicted.size:
        flat_idx = idx
        vector_size = predicted.shape[1]
        
        diff = predicted.flat[flat_idx] - target.flat[flat_idx]
        gradient.flat[flat_idx] = 2.0 * diff / vector_size

def launch_mse_loss(predicted, target, loss_output):
    """
    Launch MSE loss computation
    """
    batch_size = predicted.shape[0]
    threads_per_block = 512
    blocks_per_grid = math.ceil(batch_size / threads_per_block)
    
    gpu_mse_loss[blocks_per_grid, threads_per_block](predicted, target, loss_output)
    cuda.synchronize()

def launch_mse_gradient(predicted, target, gradient):
    """
    Launch MSE gradient computation
    """
    threads_per_block = 512
    blocks_per_grid = math.ceil(predicted.size / threads_per_block)
    
    gpu_mse_gradient[blocks_per_grid, threads_per_block](predicted, target, gradient)
    cuda.synchronize() 