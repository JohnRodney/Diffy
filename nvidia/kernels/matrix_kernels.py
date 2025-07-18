#!/usr/bin/env python3
"""
CUDA kernels for matrix operations
Designed for maximum SM utilization across modern GPUs
"""

import numba
from numba import cuda
import math

# GPU Hardware Configuration Constants
STREAMING_MULTIPROCESSOR_COUNT = 170
OPTIMAL_THREADS_PER_BLOCK = 512
MIN_BLOCKS_PER_SM = 4
TARGET_BLOCKS = STREAMING_MULTIPROCESSOR_COUNT * MIN_BLOCKS_PER_SM

# Neural Network Constants
LEAKY_RELU_DEFAULT_ALPHA = 0.01
MSE_GRADIENT_MULTIPLIER = 2.0
SHARED_MEMORY_SIZE = 512

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
        matrix_val = matrix[row, col]
        bias_val = bias[col]
        output[row, col] = matrix_val + bias_val

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
        gradient.flat[flat_idx] = MSE_GRADIENT_MULTIPLIER * diff / batch_size

@cuda.jit
def gpu_cosine_similarity_loss(predicted, target, loss_output):
    """
    Cosine similarity loss: 1 - cos_sim
    Loss per sample, computed across vector dimensions
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.blockIdx.x
    
    if sample_idx < batch_size:
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
        
        if pred_norm > 1e-8 and target_norm > 1e-8:
            cos_sim = dot_product / (pred_norm * target_norm)
            loss_output[sample_idx] = 1.0 - cos_sim
        else:
            loss_output[sample_idx] = 1.0

@cuda.jit
def gpu_cosine_similarity_gradient(predicted, target, gradient):
    """
    Cosine similarity gradient: -(target - predicted * cos_sim) / (|predicted| * |target|)
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    sample_idx = cuda.blockIdx.x
    dim_idx = cuda.threadIdx.x
    
    if sample_idx < batch_size and dim_idx < vector_size:
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
        
        if pred_norm > 1e-8 and target_norm > 1e-8:
            cos_sim = dot_product / (pred_norm * target_norm)
            norm_product = pred_norm * target_norm
            
            target_val = target[sample_idx, dim_idx]
            pred_val = predicted[sample_idx, dim_idx]
            
            gradient[sample_idx, dim_idx] = -(target_val - pred_val * cos_sim) / norm_product
        else:
            gradient[sample_idx, dim_idx] = 0.0

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
    
    if idx < A.size:
        flat_idx = idx
        val = A.flat[flat_idx]
        C.flat[flat_idx] = max(LEAKY_RELU_DEFAULT_ALPHA * val, val)

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
    Reduction sum with shared memory
    """
    idx = cuda.grid(1)
    tid = cuda.threadIdx.x
    
    shared_sum = cuda.shared.array(SHARED_MEMORY_SIZE, numba.float32, ndim=1)
    
    if idx < A.size:
        shared_sum[tid] = A.flat[idx]
    else:
        shared_sum[tid] = numba.float32(0.0)
    
    cuda.syncthreads()
    
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared_sum[tid] += shared_sum[tid + stride]
        cuda.syncthreads()
        stride //= 2
    
    if tid == 0:
        cuda.atomic.add(result, 0, shared_sum[0])



# RTX 5090 Optimized Helper Functions
def launch_matrix_multiply(A, B, C):
    """
    Launch matrix multiplication for maximum GPU utilization
    """
    M, N = C.shape
    
    threads_per_block = (32, 16)
    
    blocks_x = math.ceil(N / threads_per_block[1])
    blocks_y = math.ceil(M / threads_per_block[0])
    total_blocks = blocks_x * blocks_y
    
    if total_blocks < TARGET_BLOCKS:
        threads_per_block = (16, 16)
        blocks_x = math.ceil(N / threads_per_block[1])
        blocks_y = math.ceil(M / threads_per_block[0])
        total_blocks = blocks_x * blocks_y
    
    blocks_per_grid = (blocks_y, blocks_x)
    
    gpu_matrix_multiply[blocks_per_grid, threads_per_block](A, B, C)
    cuda.synchronize()

def launch_elementwise(A, B, C, operation_type='add'):
    """
    Launch elementwise operations for maximum GPU utilization
    """
    total_elements = A.size
    
    threads_per_block = OPTIMAL_THREADS_PER_BLOCK
    
    blocks_needed = math.ceil(total_elements / threads_per_block)
    
    blocks_needed = max(blocks_needed, TARGET_BLOCKS)
    
    if operation_type == 'add':
        gpu_elementwise_add[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'multiply':
        gpu_elementwise_multiply[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'activate':
        gpu_elementwise_activate[blocks_needed, threads_per_block](A, C)
    elif operation_type == 'copy':
        gpu_copy_array[blocks_needed, threads_per_block](A, C)
    elif operation_type == 'decay':
        gpu_elementwise_decay[blocks_needed, threads_per_block](A, C, 0.0001)
    elif operation_type == 'mse_loss':
        gpu_mse_loss[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'add_bias':
        if len(A.shape) == 2:
            threads_per_block_2d = (32, 16)
            blocks_x = math.ceil(A.shape[1] / threads_per_block_2d[1])
            blocks_y = math.ceil(A.shape[0] / threads_per_block_2d[0])
            blocks_per_grid = (blocks_y, blocks_x)
            gpu_add_bias[blocks_per_grid, threads_per_block_2d](A, B, C)
        else:
            gpu_elementwise_add[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'mse_gradient':
        gpu_mse_gradient[blocks_needed, threads_per_block](A, B, C)
    elif operation_type == 'gradient_clip':
        gpu_gradient_clip[blocks_needed, threads_per_block](A, 1.0)
    elif operation_type == 'update_weights':
        gpu_update_weights[blocks_needed, threads_per_block](A, B, 0.001)
    elif operation_type == 'activate_derivative':
        gpu_leaky_relu_backward[blocks_needed, threads_per_block](A, B, C, LEAKY_RELU_DEFAULT_ALPHA)
    elif operation_type == 'cosine_loss':
        launch_cosine_similarity_loss(A, B, C)
        return
    elif operation_type == 'cosine_gradient':
        launch_cosine_similarity_gradient(A, B, C)
        return
    
    cuda.synchronize()

def launch_cosine_similarity_loss(predicted, target, loss_output):
    """
    Launch cosine similarity loss computation for maximum GPU utilization
    predicted: (batch_size, vector_size)
    target: (batch_size, vector_size)  
    loss_output: (batch_size,)
    """
    batch_size = predicted.shape[0]
    
    threads_per_block = 1
    blocks_per_grid = batch_size
    
    blocks_per_grid = max(blocks_per_grid, TARGET_BLOCKS // 4)
    
    gpu_cosine_similarity_loss[blocks_per_grid, threads_per_block](predicted, target, loss_output)
    cuda.synchronize()

def launch_cosine_similarity_gradient(predicted, target, gradient):
    """
    Launch cosine similarity gradient computation for maximum GPU utilization
    predicted: (batch_size, vector_size)
    target: (batch_size, vector_size)
    gradient: (batch_size, vector_size)
    """
    batch_size = predicted.shape[0]
    vector_size = predicted.shape[1]
    
    threads_per_block = min(vector_size, OPTIMAL_THREADS_PER_BLOCK)
    blocks_per_grid = batch_size
    
    blocks_per_grid = max(blocks_per_grid, TARGET_BLOCKS // 4)
    
    gpu_cosine_similarity_gradient[blocks_per_grid, threads_per_block](predicted, target, gradient)
    cuda.synchronize()