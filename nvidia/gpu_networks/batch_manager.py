#!/usr/bin/env python3
"""
GPU Batch Manager for GPU Training
Handles proper mini-batching to maximize GPU utilization
"""

import numpy as np
from numba import cuda
import math


class GPUBatchManager:
    """
    Manages mini-batch creation for GPU training
    Ensures proper batch sizes for GPU utilization
    """
    
    def __init__(self, batch_size, dataset_size, vector_length):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.vector_length = vector_length
        
        # Calculate batches per epoch
        self.batches_per_epoch = math.ceil(dataset_size / batch_size)
        
        # Pre-allocate GPU memory for batches
        self.gpu_batch_data = cuda.to_device(np.zeros((batch_size, vector_length), dtype=np.float32))
        self.gpu_batch_targets = cuda.to_device(np.zeros((batch_size, vector_length), dtype=np.float32))
        
    def create_mini_batches(self, full_dataset):
        """
        Create mini-batches from the full dataset
        Returns: list of batch indices for each mini-batch
        """
        # Shuffle indices for each epoch
        shuffled_indices = np.random.permutation(self.dataset_size)
        
        batches = []
        for i in range(self.batches_per_epoch):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.dataset_size)
            
            # Get indices for this batch
            batch_indices = shuffled_indices[start_idx:end_idx]
            
            # If batch is smaller than batch_size, pad with random samples
            if len(batch_indices) < self.batch_size:
                # Pad with random samples from the dataset
                padding_needed = self.batch_size - len(batch_indices)
                padding_indices = np.random.choice(self.dataset_size, padding_needed, replace=True)
                batch_indices = np.concatenate([batch_indices, padding_indices])
            
            batches.append(batch_indices)
        
        return batches
    
    def prepare_gpu_batch(self, dataset_array, batch_indices):
        """
        Prepare a batch on GPU memory
        Returns: GPU arrays for batch data and targets
        """
        # Create batch data on CPU first
        batch_data = dataset_array[batch_indices]
        
        # For autoencoder, targets are the same as inputs
        batch_targets = batch_data.copy()
        
        # Copy to pre-allocated GPU memory
        self.gpu_batch_data[:batch_data.shape[0]] = cuda.to_device(batch_data)
        self.gpu_batch_targets[:batch_targets.shape[0]] = cuda.to_device(batch_targets)
        
        return self.gpu_batch_data, self.gpu_batch_targets
    
    def get_effective_batch_size(self):
        """
        Get the effective batch size (always the specified batch_size due to padding)
        """
        return self.batch_size
    
    def get_batches_per_epoch(self):
        """
        Get number of batches per epoch
        """
        return self.batches_per_epoch
    
    def log_batch_info(self, batch_idx, batch_data_gpu):
        """
        Log information about the current batch
        """
        pass
    
    def estimate_blocks_for_batch(self, batch_size):
        """
        Estimate GPU blocks needed for matrix operations with this batch size
        """
        # For typical autoencoder layer (batch_size x vector_length) @ (vector_length x hidden_size)
        # Using (16, 16) threads per block as fallback
        threads_per_block = (16, 16)
        
        # Estimate for a typical hidden layer size (e.g., 320)
        hidden_size = 320
        
        blocks_x = math.ceil(hidden_size / threads_per_block[1])
        blocks_y = math.ceil(batch_size / threads_per_block[0])
        total_blocks = blocks_x * blocks_y
        
        return total_blocks


def create_batch_manager(batch_size, dataset_size, vector_length):
    """
    Factory function to create a batch manager optimized for GPU training
    """
    return GPUBatchManager(batch_size, dataset_size, vector_length) 