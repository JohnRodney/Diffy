#!/usr/bin/env python3
"""
RTX 5090 Optimized CUDA kernels for matrix operations
Designed for maximum SM utilization across 170 streaming multiprocessors
"""

import numba
from numba import cuda
import numpy as np
import math

# RTX 5090 Configuration Constants
RTX5090_SM_COUNT = 170
OPTIMAL_THREADS_PER_BLOCK = 512  # Sweet spot for RTX 5090
MIN_BLOCKS_PER_SM = 4  # Minimum blocks per SM for good occupancy
TARGET_BLOCKS = RTX5090_SM_COUNT * MIN_BLOCKS_PER_SM  # 680 blocks minimum

@cuda.jit
def gpu_matrix_multiply(A, B, C):
    """
    GPU matrix multiplication: C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    Optimized for RTX 5090 with proper thread block sizing
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
    Optimized for high parallelism
    """
    row, col = cuda.grid(2)
    
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # Explicitly cast to ensure scalar arithmetic
        matrix_val = matrix[row, col]
        bias_val = bias[col]
        output[row, col] = matrix_val + bias_val

@cuda.jit
def gpu_leaky_relu_forward(input_data, output, alpha):
    """
    Leaky ReLU activation: output = max(alpha * input, input)
    Massively parallel version for RTX 5090
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

@cuda.jit
def gpu_elementwise_add(A, B, C):
    """
    Elementwise addition: C = A + B
    """
    idx = cuda.grid(1)
    
    if idx < A.size:
        flat_idx = idx
        C.flat[flat_idx] = A.flat[flat_idx] + B.flat[flat_idx]

@cuda.jit
def gpu_elementwise_multiply(A, B, C):
    """
    Elementwise multiplication: C = A * B
    """
    idx = cuda.grid(1)
    
    if idx < A.size:
        flat_idx = idx
        C.flat[flat_idx] = A.flat[flat_idx] * B.flat[flat_idx]

@cuda.jit
def gpu_elementwise_activate(A, C):
    """
    Elementwise activation (LeakyReLU): C = max(0.01 * A, A)
    """
    idx = cuda.grid(1)
    alpha = 0.01
    
    if idx < A.size:
        flat_idx = idx
        val = A.flat[flat_idx]
        C.flat[flat_idx] = max(alpha * val, val)

@cuda.jit
def gpu_elementwise_decay(A, C, decay_factor):
    """
    Elementwise weight decay: C = A * (1 - decay_factor)
    """
    idx = cuda.grid(1)
    
    if idx < A.size:
        flat_idx = idx
        C.flat[flat_idx] = A.flat[flat_idx] * (1.0 - decay_factor)

@cuda.jit
def gpu_reduce_sum(A, result):
    """
    Reduction sum with shared memory optimization
    """
    idx = cuda.grid(1)
    tid = cuda.threadIdx.x
    
    # Shared memory for block-level reduction
    shared_sum = cuda.shared.array(512, numba.float32)
    
    # Load data into shared memory
    if idx < A.size:
        shared_sum[tid] = A.flat[idx]
    else:
        shared_sum[tid] = 0.0
    
    cuda.syncthreads()
    
    # Reduction within block
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared_sum[tid] += shared_sum[tid + stride]
        cuda.syncthreads()
        stride //= 2
    
    # Write block result
    if tid == 0:
        cuda.atomic.add(result, 0, shared_sum[0])

# RTX 5090 Optimized Helper Functions
def launch_matrix_multiply_rtx5090(A, B, C):
    """
    Launch matrix multiplication optimized for RTX 5090
    Ensures >680 blocks across 170 SMs for maximum utilization
    """
    M, N = C.shape
    
    # Calculate grid dimensions for RTX 5090 optimization
    # Use 32x16 threads per block = 512 threads (optimal for RTX 5090)
    threads_per_block = (32, 16)
    
    # Calculate blocks needed
    blocks_x = math.ceil(N / threads_per_block[1])  # columns
    blocks_y = math.ceil(M / threads_per_block[0])  # rows
    total_blocks = blocks_x * blocks_y
    
    # Ensure we have at least 680 blocks for RTX 5090 (4 blocks per SM)
    if total_blocks < TARGET_BLOCKS:
        # Adjust thread block size to create more blocks
        threads_per_block = (16, 16)  # 256 threads per block
        blocks_x = math.ceil(N / threads_per_block[1])
        blocks_y = math.ceil(M / threads_per_block[0])
        total_blocks = blocks_x * blocks_y
    
    blocks_per_grid = (blocks_y, blocks_x)
    
    print(f"🚀 RTX5090 Matrix Multiply: {total_blocks} blocks across {RTX5090_SM_COUNT} SMs")
    print(f"   Grid: {blocks_per_grid}, Threads: {threads_per_block}")
    
    # Launch kernel
    gpu_matrix_multiply[blocks_per_grid, threads_per_block](A, B, C)
    cuda.synchronize()

def launch_elementwise_rtx5090(A, B, C, operation_type='add'):
    """
    Launch elementwise operations optimized for RTX 5090
    Creates enough blocks to saturate all 170 SMs
    """
    total_elements = A.size
    
    # Use 512 threads per block (optimal for RTX 5090)
    threads_per_block = OPTIMAL_THREADS_PER_BLOCK
    
    # Calculate blocks needed based on actual work
    blocks_needed = math.ceil(total_elements / threads_per_block)
    
    # Only enforce minimum block count for larger operations
    if total_elements > 100000:  # Only for large operations
        blocks_needed = max(blocks_needed, TARGET_BLOCKS)
    
    print(f"🚀 RTX5090 Elementwise {operation_type}: {blocks_needed} blocks, {total_elements} elements")
    print(f"   Threads per block: {threads_per_block}, SM utilization: {min(100, blocks_needed/RTX5090_SM_COUNT*100):.1f}%")
    
    # Launch appropriate kernel
    if operation_type == 'add':
        gpu_elementwise_add[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'multiply':
        gpu_elementwise_multiply[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'activate':
        gpu_elementwise_activate[blocks_needed, threads_per_block](A, C)
    elif operation_type == 'copy':
        gpu_copy_array[blocks_needed, threads_per_block](A, C)
    elif operation_type == 'decay':
        # Weight decay operation: A *= 0.9999 (simple decay)
        gpu_elementwise_decay[blocks_needed, threads_per_block](A, C, 0.0001)
    elif operation_type == 'mse_loss':
        # MSE loss computation: (A - B)^2
        gpu_mse_loss[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'add_bias':
        # Special handling for bias addition (2D operation)
        if len(A.shape) == 2:  # Matrix + bias vector
            threads_per_block_2d = (32, 16)
            blocks_x = math.ceil(A.shape[1] / threads_per_block_2d[1])
            blocks_y = math.ceil(A.shape[0] / threads_per_block_2d[0])
            blocks_per_grid = (blocks_y, blocks_x)
            total_blocks_2d = blocks_x * blocks_y
            print(f"   2D operation: {total_blocks_2d} blocks ({blocks_y}x{blocks_x} grid)")
            gpu_add_bias[blocks_per_grid, threads_per_block_2d](A, B, C)
        else:
            gpu_elementwise_add[blocks_needed, threads_per_block](A, B, C)
    
    cuda.synchronize()

# Validation function
def validate_rtx5090_utilization(blocks_per_grid, threads_per_block):
    """
    Validate that our configuration will properly utilize RTX 5090
    """
    if isinstance(blocks_per_grid, tuple):
        total_blocks = blocks_per_grid[0] * blocks_per_grid[1]
    else:
        total_blocks = blocks_per_grid
    
    total_threads = total_blocks * (threads_per_block[0] * threads_per_block[1] if isinstance(threads_per_block, tuple) else threads_per_block)
    
    sm_utilization = min(total_blocks / RTX5090_SM_COUNT, 1.0)
    
    print(f"📊 RTX5090 Utilization Analysis:")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Total threads: {total_threads}")
    print(f"   SM utilization: {sm_utilization:.1%}")
    print(f"   Blocks per SM: {total_blocks / RTX5090_SM_COUNT:.1f}")
    
    if sm_utilization < 0.8:
        print(f"⚠️  WARNING: Low SM utilization {sm_utilization:.1%} - expect poor GPU performance")
    else:
        print(f"✅ Good SM utilization {sm_utilization:.1%} - should see high GPU usage")
    
    return sm_utilization 