#!/usr/bin/env python3
"""
Pure numba CUDA kernels for matrix operations
No external libraries - just numba.cuda.jit
"""

import numba
from numba import cuda
import numpy as np
import math

@cuda.jit
def gpu_matrix_multiply(A, B, C):
    """
    GPU matrix multiplication: C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    """
    row, col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        temp = 0.0
        for k in range(A.shape[1]):
            temp += A[row, k] * B[k, col]
        C[row, col] = temp

@cuda.jit
def gpu_add_bias(matrix, bias, output):
    """
    Add bias to each row: output = matrix + bias
    matrix: (batch_size, features), bias: (features,)
    """
    row, col = cuda.grid(2)
    
    if row < matrix.shape[0] and col < matrix.shape[1]:
        output[row, col] = matrix[row, col] + bias[col]

@cuda.jit
def gpu_leaky_relu_forward(input_data, output, alpha):
    """
    Leaky ReLU activation: output = max(alpha * input, input)
    """
    idx = cuda.grid(1)
    
    if idx < input_data.size:
        flat_idx = idx
        val = input_data.flat[flat_idx]
        output.flat[flat_idx] = max(alpha * val, val)

@cuda.jit
def gpu_leaky_relu_backward(input_data, grad_output, grad_input, alpha):
    """
    Leaky ReLU backward: grad_input = grad_output * (input > 0 ? 1 : alpha)
    """
    idx = cuda.grid(1)
    
    if idx < input_data.size:
        flat_idx = idx
        val = input_data.flat[flat_idx]
        derivative = 1.0 if val > 0 else alpha
        grad_input.flat[flat_idx] = grad_output.flat[flat_idx] * derivative

@cuda.jit
def gpu_mse_loss(predicted, target, loss_output):
    """
    Mean squared error loss
    """
    idx = cuda.grid(1)
    
    if idx < predicted.size:
        flat_idx = idx
        diff = predicted.flat[flat_idx] - target.flat[flat_idx]
        loss_output.flat[flat_idx] = diff * diff

@cuda.jit
def gpu_mse_gradient(predicted, target, gradient):
    """
    MSE gradient: 2 * (predicted - target) / batch_size
    """
    idx = cuda.grid(1)
    batch_size = predicted.shape[0]
    
    if idx < predicted.size:
        flat_idx = idx
        diff = predicted.flat[flat_idx] - target.flat[flat_idx]
        gradient.flat[flat_idx] = 2.0 * diff / batch_size

@cuda.jit
def gpu_gradient_clip(gradient, max_norm):
    """
    Clip gradients to max_norm
    """
    idx = cuda.grid(1)
    
    if idx < gradient.size:
        flat_idx = idx
        val = gradient.flat[flat_idx]
        if val > max_norm:
            gradient.flat[flat_idx] = max_norm
        elif val < -max_norm:
            gradient.flat[flat_idx] = -max_norm

@cuda.jit
def gpu_update_weights(weights, gradients, learning_rate):
    """
    Update weights: weights = weights - learning_rate * gradients
    """
    idx = cuda.grid(1)
    
    if idx < weights.size:
        flat_idx = idx
        weights.flat[flat_idx] -= learning_rate * gradients.flat[flat_idx]

@cuda.jit
def gpu_copy_array(source, dest):
    """
    Copy array on GPU
    """
    idx = cuda.grid(1)
    
    if idx < source.size:
        flat_idx = idx
        dest.flat[flat_idx] = source.flat[flat_idx]

# Helper functions for kernel launching
def launch_matrix_multiply(A_gpu, B_gpu, C_gpu):
    """Launch matrix multiplication kernel with optimal grid size"""
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(C_gpu.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(C_gpu.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    gpu_matrix_multiply[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)

def launch_elementwise_kernel(kernel_func, *args):
    """Launch elementwise kernel with optimal grid size"""
    # Assume first argument is the array to determine size
    array_size = args[0].size
    threads_per_block = 256
    blocks_per_grid = math.ceil(array_size / threads_per_block)
    
    kernel_func[blocks_per_grid, threads_per_block](*args)

def launch_2d_kernel(kernel_func, array_2d, *args):
    """Launch 2D kernel with optimal grid size"""
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(array_2d.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(array_2d.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    kernel_func[blocks_per_grid, threads_per_block](array_2d, *args) 