#!/usr/bin/env python3
"""
Embedding lookup CUDA kernels
Standard approach: token_id -> embedding_table[token_id]
"""

import numba
from numba import cuda
import numpy as np
import math

@cuda.jit
def gpu_embedding_lookup(token_ids, embedding_table, output):
    """
    Embedding lookup: output[i] = embedding_table[token_ids[i]]
    
    token_ids: (batch_size,) - token IDs to look up
    embedding_table: (vocab_size, embedding_dim) - learnable embedding table
    output: (batch_size, embedding_dim) - output vectors
    """
    idx = cuda.grid(1)
    
    if idx < token_ids.size:
        token_id = int(token_ids[idx])
        embedding_dim = embedding_table.shape[1]
        
        # Copy embedding vector to output
        for dim in range(embedding_dim):
            output[idx, dim] = embedding_table[token_id, dim]

@cuda.jit
def gpu_embedding_gradient(token_ids, grad_output, embedding_grad):
    """
    Accumulate gradients for embedding table
    
    token_ids: (batch_size,) - token IDs that were looked up
    grad_output: (batch_size, embedding_dim) - gradients from next layer
    embedding_grad: (vocab_size, embedding_dim) - accumulated gradients for embedding table
    """
    idx = cuda.grid(1)
    
    if idx < token_ids.size:
        token_id = int(token_ids[idx])
        embedding_dim = grad_output.shape[1]
        
        # Accumulate gradients for this token's embedding
        for dim in range(embedding_dim):
            embedding_grad[token_id, dim] += grad_output[idx, dim]

def launch_embedding_lookup(token_ids, embedding_table, output):
    """
    Launch embedding lookup kernel
    """
    threads_per_block = 512
    blocks_per_grid = math.ceil(token_ids.size / threads_per_block)
    
    gpu_embedding_lookup[blocks_per_grid, threads_per_block](token_ids, embedding_table, output)
    cuda.synchronize()

def launch_embedding_gradient(token_ids, grad_output, embedding_grad):
    """
    Launch embedding gradient accumulation kernel
    """
    threads_per_block = 512
    blocks_per_grid = math.ceil(token_ids.size / threads_per_block)
    
    gpu_embedding_gradient[blocks_per_grid, threads_per_block](token_ids, grad_output, embedding_grad)
    cuda.synchronize() 