#!/usr/bin/env python3
"""
GPU Text Autoencoder using numba-cuda kernels
All operations happen on GPU memory - no CPU transfers during training
"""

import numpy as np
import math
from numba import cuda
from kernels.matrix_kernels import (
    launch_matrix_multiply,
    gpu_gradient_clip,
    gpu_update_weights,
    gpu_leaky_relu_backward,
    gpu_add_bias,
    gpu_leaky_relu_forward,
    gpu_copy_array
)
from kernels.loss_kernels import (
    launch_mse_loss,
    launch_mse_gradient
)
from kernels.embedding_kernels import (
    launch_embedding_lookup,
    launch_embedding_gradient
)
from kernels.transpose_kernels import create_transpose_gpu
from gpu_networks.pca_initialization import compute_pca_weights

def calculate_launch_config(total_elements, threads_per_block=512):
    """Calculate CUDA launch configuration for 1D kernels"""
    blocks_needed = math.ceil(total_elements / threads_per_block)
    return blocks_needed, threads_per_block

def calculate_launch_config_2d(shape, threads_per_block=(16, 16)):
    """Calculate CUDA launch configuration for 2D kernels"""
    blocks_x = math.ceil(shape[1] / threads_per_block[1])
    blocks_y = math.ceil(shape[0] / threads_per_block[0])
    return (blocks_y, blocks_x), threads_per_block

class GPUTextAutoencoder:
    """
    GPU-resident text autoencoder for fast training
    All weights and computations stay on GPU
    """
    
    def __init__(self, vector_length, hidden_layer_count, bottleneck_size, alpha=0.01):
        self.vector_length = vector_length
        self.hidden_layer_count = hidden_layer_count
        self.bottleneck_size = bottleneck_size
        self.alpha = alpha
        
        # Training hyperparameters (set once, used throughout training)
        self.learning_rate = 0.0001
        self.grad_clip_norm = 1.0
        
        # Internal dataset storage
        self.dataset_gpu: cuda.DeviceNDArray | None = None
        self.dataset_size = 0
        
        # Embedding table for token-based input
        self.embedding_table: cuda.DeviceNDArray | None = None
        self.use_embeddings = False
        
        # GPU memory buffers (no class weight storage)
        self.weight_buffers = []  # Direct GPU weight buffers
        self.bias_buffers = []    # Direct GPU bias buffers
        self.layer_outputs = []
        self.layer_inputs = []
        self.gradients = []
        
        self.layer_sizes = self._calculate_layer_sizes()
        self._initialize_weights_in_gpu()
        self._allocate_gpu_memory()
        
        # Log model information
        param_count = self._count_parameters()
        print(f"GPUTextAutoencoder initialized: {param_count:,} parameters")
    
    def _calculate_layer_sizes(self):
        """Calculate gradual stepdown layer sizes using geometric progression"""
        sizes = [self.vector_length]
        
        if self.hidden_layer_count == 0:
            # Direct connection: input -> bottleneck -> output
            sizes.append(self.bottleneck_size)
            sizes.append(self.vector_length)
        else:
            # Geometric stepdown for smoother compression
            ratio = (self.bottleneck_size / self.vector_length) ** (1.0 / (self.hidden_layer_count + 1))
            
            # Encoder layers: gradual geometric reduction
        current_size = self.vector_length
        for i in range(self.hidden_layer_count):
            current_size = max(int(current_size * ratio), self.bottleneck_size)
            sizes.append(current_size)
            if current_size == self.bottleneck_size:
                break
        
        # Ensure we end at bottleneck
        if sizes[-1] != self.bottleneck_size:
            sizes.append(self.bottleneck_size)
        
        # Decoder layers (mirror of encoder, excluding input and bottleneck)
            encoder_sizes = sizes[1:-1]  # Get middle layers
            encoder_sizes.reverse()      # Reverse for decoder
        
        # Add decoder layers: bottleneck -> ... -> output
        sizes.extend(encoder_sizes)
        sizes.append(self.vector_length)
        
        return sizes
    
    def _initialize_weights_in_gpu(self):
        """Initialize weights directly in GPU memory buffers"""
        
        # Try PCA initialization if training data is available
        use_pca = hasattr(self, 'training_data') and self.training_data is not None
        encoder_weights = None
        decoder_weights = None
        
        if use_pca:
            encoder_weights, decoder_weights = compute_pca_weights(
                self.training_data, self.bottleneck_size
            )
        
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            # Use PCA weights for encoder (first) and decoder (last) layers
            if use_pca and i == 0 and encoder_weights is not None:  # Encoder layer
                weights_cpu = encoder_weights.copy()
            elif use_pca and i == len(self.layer_sizes) - 2 and decoder_weights is not None:  # Decoder layer
                weights_cpu = decoder_weights.copy()
            else:  # Middle layers or fallback to random
                # Xavier initialization on CPU first (for proper random distribution)
            limit = math.sqrt(6.0 / (input_size + output_size))
            weights_cpu = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
            
            biases_cpu = np.zeros(output_size, dtype=np.float32)
            
            # Create GPU buffers and initialize with CPU data
            weight_buffer = cuda.to_device(weights_cpu)
            bias_buffer = cuda.to_device(biases_cpu)
            
            self.weight_buffers.append(weight_buffer)
            self.bias_buffers.append(bias_buffer)
    
    def _allocate_gpu_memory(self):
        """Pre-allocate GPU memory for forward/backward passes"""
        max_batch_size = 4096  # Scaled for RTX 5090
        
        # Allocate memory for each layer's outputs
        for layer_size in self.layer_sizes:
            layer_output = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            layer_input = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            gradient = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            
            self.layer_outputs.append(layer_output)
            self.layer_inputs.append(layer_input)
            self.gradients.append(gradient)
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent memory leaks on shutdown"""
        try:
            # Clear all GPU arrays on shutdown
            self.layer_outputs.clear()
            self.layer_inputs.clear()
            self.gradients.clear()
            self.weight_buffers.clear()
            self.bias_buffers.clear()
            self.dataset_gpu = None
            self.dataset_size = 0
            
            # Force garbage collection
            cuda.current_context().synchronize()
        except Exception as e:
            pass
    
    def _count_parameters(self):
        """Count total parameters in the model"""
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            total += input_size * output_size + output_size  # weights + biases
        return total
    
    def forward(self, input_batch):
        """
        Forward pass entirely on GPU using RTX 5090 optimized kernels
        input_batch_gpu: GPU array of shape (batch_size, vector_length)
        returns: GPU array of shape (batch_size, vector_length)
        """
        # Ensure we're on the right device
        cuda.select_device(0)
        
        current_output = input_batch
        batch_size = input_batch.shape[0]
        
        # Store input for backward pass
        blocks, threads = calculate_launch_config(current_output.size)
        gpu_copy_array[blocks, threads](current_output, self.layer_inputs[0][:batch_size])
        
        # Forward through all layers
        for i in range(len(self.weight_buffers)):
            # Linear transformation: output = input @ weights + bias
            temp_output = self.layer_outputs[i+1][:batch_size]
            
            # Matrix multiplication
            launch_matrix_multiply(current_output, self.weight_buffers[i], temp_output)
            
            # Add bias
            blocks_2d, threads_2d = calculate_launch_config_2d(temp_output.shape)
            gpu_add_bias[blocks_2d, threads_2d](temp_output, self.bias_buffers[i], temp_output)
            
            # Apply activation (except for last layer)
            if i < len(self.weight_buffers) - 1:  # Not output layer
                blocks, threads = calculate_launch_config(temp_output.size)
                gpu_leaky_relu_forward[blocks, threads](temp_output, temp_output, self.alpha)
            
            # Store for backward pass
            if i < len(self.layer_inputs) - 1:
                blocks, threads = calculate_launch_config(temp_output.size)
                gpu_copy_array[blocks, threads](temp_output, self.layer_inputs[i + 1][:batch_size])
            
            current_output = temp_output
        
        # Single synchronization at the end
        cuda.synchronize()
        
        return current_output
    
    def backward(self, input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm, batch_indices=None):
        """
        Simple backward pass with MSE loss
        """
        batch_size = input_batch_gpu.shape[0]
        max_batch_size = self.layer_outputs[0].shape[0]
        
        # Safety check to prevent illegal memory access
        if batch_size > max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds pre-allocated memory {max_batch_size}")
        
        # Forward pass to get outputs (uses pre-allocated memory)
        final_output = self.forward(input_batch_gpu)
        
        # Compute initial gradient (Simple MSE for reconstruction)
        current_grad = self.gradients[-1][:batch_size]
        launch_mse_gradient(final_output, target_batch_gpu, current_grad)
        
        # Backward through all layers
        for i in reversed(range(len(self.weight_buffers))):
            layer_input = self.layer_inputs[i][:batch_size]
            
            # Compute weight gradients
            weight_grad = cuda.device_array_like(self.weight_buffers[i])
            # Use optimized transpose instead of .T operation
            layer_input_T = create_transpose_gpu(layer_input)
            launch_matrix_multiply(layer_input_T, current_grad, weight_grad)
            
            # Compute bias gradients (simplified - sum over batch dimension)
            bias_grad = cuda.device_array_like(self.bias_buffers[i])
            
            # Clip gradients
            blocks, threads = calculate_launch_config(weight_grad.size)
            gpu_gradient_clip[blocks, threads](weight_grad, grad_clip_norm)
            blocks, threads = calculate_launch_config(bias_grad.size)
            gpu_gradient_clip[blocks, threads](bias_grad, grad_clip_norm)
            
            # Update weights
            blocks, threads = calculate_launch_config(self.weight_buffers[i].size)
            gpu_update_weights[blocks, threads](self.weight_buffers[i], weight_grad, learning_rate)
            blocks, threads = calculate_launch_config(self.bias_buffers[i].size)
            gpu_update_weights[blocks, threads](self.bias_buffers[i], bias_grad, learning_rate)
            
            # Compute gradient for previous layer
            if i > 0:
                prev_grad = self.gradients[i][:batch_size]  # Fixed: use correct gradient buffer
                # Use optimized transpose instead of .T operation
                weights_T = create_transpose_gpu(self.weight_buffers[i])
                launch_matrix_multiply(current_grad, weights_T, prev_grad)
                
                # Apply activation derivative
                layer_input_prev = self.layer_inputs[i][:batch_size]  # Fixed: use correct layer input
                blocks, threads = calculate_launch_config(layer_input_prev.size)
                gpu_leaky_relu_backward[blocks, threads](layer_input_prev, prev_grad, prev_grad, self.alpha)
                
                current_grad = prev_grad
    
        # Update embedding table if using embeddings
        if self.use_embeddings and batch_indices is not None and self.embedding_table is not None and self.dataset_gpu is not None:
            # Get token IDs for this batch
            batch_token_ids = cuda.device_array((batch_size,), dtype=np.int32)
            for i, idx in enumerate(batch_indices):
                batch_token_ids[i] = self.dataset_gpu[idx]
            
            # Get gradients for the first layer (input layer)
            input_grad = self.gradients[0][:batch_size]
            
            # Update embedding table
            embedding_grad = cuda.device_array_like(self.embedding_table)
            launch_embedding_gradient(batch_token_ids, input_grad, embedding_grad)
            
            # Apply embedding updates
            blocks, threads = calculate_launch_config(self.embedding_table.size)
            gpu_update_weights[blocks, threads](self.embedding_table, embedding_grad, learning_rate)
    
        # Single synchronization at the end
        cuda.synchronize()
    
    def set_learning_rate(self, learning_rate):
        """Set learning rate for training"""
        self.learning_rate = learning_rate
        
    def set_grad_clip_norm(self, grad_clip_norm):
        """Set gradient clipping norm for training"""
        self.grad_clip_norm = grad_clip_norm
    

    
    def set_training_data_for_pca(self, training_data):
        """Set training data for PCA-based weight initialization"""
        self.training_data = training_data
    
    def create_embedding_table(self, vocab_size, embedding_dim):
        """Create and initialize embedding table on GPU"""
        # Xavier initialization for embedding table
        limit = math.sqrt(6.0 / (vocab_size + embedding_dim))
        embedding_init = np.random.uniform(-limit, limit, (vocab_size, embedding_dim)).astype(np.float32)
        
        # Transfer to GPU
        self.embedding_table = cuda.to_device(embedding_init)
        self.use_embeddings = True
    
    def train_batch(self, batch_indices):
        """
        Train on a batch using internal dataset and hyperparameters
        """
        if self.dataset_gpu is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get batch from GPU dataset
        if self.use_embeddings:
            # Look up embeddings for token IDs
            batch_token_ids = cuda.device_array((len(batch_indices),), dtype=np.int32)
            for i, idx in enumerate(batch_indices):
                batch_token_ids[i] = self.dataset_gpu[idx]
            
            # Convert token IDs to embeddings
            batch_gpu = cuda.device_array((len(batch_indices), self.vector_length), dtype=np.float32)
            launch_embedding_lookup(batch_token_ids, self.embedding_table, batch_gpu)
            else:
            # Direct vector input
            batch_gpu = cuda.device_array((len(batch_indices), self.dataset_gpu.shape[1]), dtype=np.float32)
            for i, idx in enumerate(batch_indices):
                batch_gpu[i] = self.dataset_gpu[idx]
        
        target_batch_gpu = batch_gpu  # Autoencoder targets = inputs
        
        # Forward and backward pass using internal hyperparameters
        self.backward(batch_gpu, target_batch_gpu, self.learning_rate, self.grad_clip_norm, batch_indices)
        
        # Compute loss if needed (simple MSE reconstruction)
        final_output = self.forward(batch_gpu)
        loss_gpu = cuda.device_array((len(batch_indices),), dtype=np.float32)
        
        launch_mse_loss(final_output, target_batch_gpu, loss_gpu)
        
        loss_cpu = loss_gpu.copy_to_host()
        return float(np.mean(loss_cpu))
    
    def infer(self, input_batch_cpu):
        """
        Inference - returns result to CPU
        """
        input_batch_gpu = cuda.to_device(input_batch_cpu.astype(np.float32))
        output_gpu = self.forward(input_batch_gpu)
        return output_gpu.copy_to_host()
    
    def export_weights(self):
        """
        Export all weights as CPU arrays for saving/analysis
        """
        weights_cpu = []
        biases_cpu = []
        
        for w_gpu, b_gpu in zip(self.weight_buffers, self.bias_buffers):
            weights_cpu.append(w_gpu.copy_to_host())
            biases_cpu.append(b_gpu.copy_to_host())
        
        return weights_cpu, biases_cpu
    
    def load_weights(self, weights_cpu, biases_cpu):
        """
        Load weights from CPU arrays
        """
        for i, (w_cpu, b_cpu) in enumerate(zip(weights_cpu, biases_cpu)):
            self.weight_buffers[i] = cuda.to_device(w_cpu.astype(np.float32))
            self.bias_buffers[i] = cuda.to_device(b_cpu.astype(np.float32))