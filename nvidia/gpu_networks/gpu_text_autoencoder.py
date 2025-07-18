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
    gpu_copy_array,
    gpu_cosine_similarity_loss,
    gpu_cosine_similarity_gradient
)
from kernels.transpose_kernels import create_transpose_gpu

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
        """Calculate the size of each layer"""
        sizes = [self.vector_length]
        
        # Encoder layers
        current_size = self.vector_length
        reduction_factor = math.ceil((self.vector_length - self.bottleneck_size) / self.hidden_layer_count)
        
        for i in range(self.hidden_layer_count):
            next_size = max(current_size - reduction_factor, self.bottleneck_size)
            sizes.append(next_size)
            current_size = next_size
            if current_size == self.bottleneck_size:
                break
        
        # Ensure we end at bottleneck
        if sizes[-1] != self.bottleneck_size:
            sizes.append(self.bottleneck_size)
        
        # Decoder layers (mirror of encoder, excluding input and bottleneck)
        encoder_sizes = sizes[1:-1]  # Get middle layers (exclude input and bottleneck)
        encoder_sizes.reverse()      # Reverse the order
        
        # Add decoder layers: bottleneck -> ... -> output
        sizes.extend(encoder_sizes)
        sizes.append(self.vector_length)  # Final output layer
        
        return sizes
    
    def _initialize_weights_in_gpu(self):
        """Initialize weights directly in GPU memory buffers"""
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
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
    
    def backward(self, input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm):
        """
        Backward pass with RTX 5090 optimized kernels and proper memory bounds checking
        """
        batch_size = input_batch_gpu.shape[0]
        max_batch_size = self.layer_outputs[0].shape[0]
        
        # Safety check to prevent illegal memory access
        if batch_size > max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds pre-allocated memory {max_batch_size}")
        
        # Forward pass to get outputs (uses pre-allocated memory)
        final_output = self.forward(input_batch_gpu)
        
        # Compute initial gradient (Cosine similarity loss)
        current_grad = self.gradients[-1][:batch_size]
        blocks_per_grid = batch_size
        threads_per_block = min(final_output.shape[1], 512)
        gpu_cosine_similarity_gradient[blocks_per_grid, threads_per_block](final_output, target_batch_gpu, current_grad)
        
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
    
        # Single synchronization at the end
        cuda.synchronize()
    
    def set_learning_rate(self, learning_rate):
        """Set learning rate for training"""
        self.learning_rate = learning_rate
        
    def set_grad_clip_norm(self, grad_clip_norm):
        """Set gradient clipping norm for training"""
        self.grad_clip_norm = grad_clip_norm
    
    def train_batch(self, batch_indices):
        """
        Train on a batch using internal dataset and hyperparameters
        """
        if self.dataset_gpu is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get batch from GPU dataset
        batch_gpu = cuda.device_array((len(batch_indices), self.dataset_gpu.shape[1]), dtype=np.float32)
        for i, idx in enumerate(batch_indices):
            batch_gpu[i] = self.dataset_gpu[idx]
        target_batch_gpu = batch_gpu  # Autoencoder targets = inputs
        
        # Forward and backward pass using internal hyperparameters
        self.backward(batch_gpu, target_batch_gpu, self.learning_rate, self.grad_clip_norm)
        
        # Compute loss if needed
        final_output = self.forward(batch_gpu)
        loss_gpu = cuda.device_array((len(batch_indices),), dtype=np.float32)
        blocks_per_grid = len(batch_indices)
        threads_per_block = 1
        gpu_cosine_similarity_loss[blocks_per_grid, threads_per_block](final_output, target_batch_gpu, loss_gpu)
        
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