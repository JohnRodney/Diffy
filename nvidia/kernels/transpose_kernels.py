#!/usr/bin/env python3
"""
GPU Matrix Transpose Kernels
Based on NVIDIA's "An Efficient Matrix Transpose in CUDA C/C++" blog post
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

Achieves ~95% of copy performance by:
1. Using shared memory to coalesce global memory access
2. Padding shared memory to avoid bank conflicts
3. Proper thread block sizing (32x8 threads, 32x32 tile)
"""

import numpy as np
import math
from numba import cuda
import numba

# Transpose Configuration
TILE_DIM = 32
BLOCK_ROWS = 8
SHARED_MEMORY_PADDING = 1

@cuda.jit
def gpu_transpose(input_matrix, output_matrix):
    """
    GPU matrix transpose kernel - avoids bank conflicts
    
    Args:
        input_matrix: Input matrix (M, N)
        output_matrix: Output matrix (N, M) - must be pre-allocated
    """
    tile = cuda.shared.array((TILE_DIM, TILE_DIM + SHARED_MEMORY_PADDING), dtype=numba.float32)
    
    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y
    width = cuda.gridDim.x * TILE_DIM
    
    for j in range(0, TILE_DIM, BLOCK_ROWS):
        if (y + j) < input_matrix.shape[0] and x < input_matrix.shape[1]:
            tile[cuda.threadIdx.y + j, cuda.threadIdx.x] = input_matrix[y + j, x]
    
    cuda.syncthreads()
    
    x = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.y
    
    for j in range(0, TILE_DIM, BLOCK_ROWS):
        if (y + j) < output_matrix.shape[0] and x < output_matrix.shape[1]:
            output_matrix[y + j, x] = tile[cuda.threadIdx.x, cuda.threadIdx.y + j]

def launch_transpose(input_matrix, output_matrix):
    """
    Launch matrix transpose with proper grid/block configuration
    
    Args:
        input_matrix: GPU array (M, N)
        output_matrix: GPU array (N, M) - must be pre-allocated
    """
    if input_matrix.shape[0] != output_matrix.shape[1] or input_matrix.shape[1] != output_matrix.shape[0]:
        raise ValueError(f"Incompatible dimensions: input {input_matrix.shape}, output {output_matrix.shape}")
    
    grid_x = math.ceil(input_matrix.shape[1] / TILE_DIM)
    grid_y = math.ceil(input_matrix.shape[0] / TILE_DIM)
    
    threads_per_block = (TILE_DIM, BLOCK_ROWS)
    blocks_per_grid = (grid_x, grid_y)
    
    gpu_transpose[blocks_per_grid, threads_per_block](input_matrix, output_matrix)
    
    cuda.synchronize()

def create_transpose_gpu(input_matrix_gpu):
    """
    Create and return transposed matrix on GPU
    
    Args:
        input_matrix_gpu: GPU array (M, N)
        
    Returns:
        GPU array (N, M) - transposed matrix
    """
    output_shape = (input_matrix_gpu.shape[1], input_matrix_gpu.shape[0])
    output_matrix_gpu = cuda.device_array(output_shape, dtype=input_matrix_gpu.dtype)
    
    launch_transpose(input_matrix_gpu, output_matrix_gpu)
    
    return output_matrix_gpu 