#!/usr/bin/env python3
"""
GPU Text Autoencoder using numba-cuda kernels
All operations happen on GPU memory - no CPU transfers during training
"""

import numpy as np
import math
from numba import cuda
from nvidia.kernels.matrix_kernels import (
    launch_matrix_multiply,
    gpu_mse_gradient,
    gpu_gradient_clip,
    gpu_update_weights,
    gpu_leaky_relu_backward,
    gpu_mse_loss,
    gpu_add_bias,
    gpu_leaky_relu_forward,
    gpu_copy_array
)
from nvidia.kernels.transpose_kernels import create_transpose_gpu

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
        
        # Explicitly select CUDA device 0 to ensure GPU execution
        cuda.select_device(0)
        
        # Calculate layer sizes
        self.layer_sizes = self._calculate_layer_sizes()
        
        # Initialize weights on GPU
        self.weights_gpu = []
        self.biases_gpu = []
        self._initialize_weights_on_gpu()
        
        # Pre-allocate GPU memory for forward/backward passes
        self.layer_outputs_gpu = []
        self.layer_inputs_gpu = []
        self.gradients_gpu = []
        self._allocate_gpu_memory()
    
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
    
    def _initialize_weights_on_gpu(self):
        """Initialize all weights and biases directly on GPU"""
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            # Xavier initialization
            limit = math.sqrt(6.0 / (input_size + output_size))
            weights_cpu = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
            biases_cpu = np.zeros(output_size, dtype=np.float32)  # 1D array, not 2D
            
            # Transfer to GPU
            weights_gpu = cuda.to_device(weights_cpu)
            biases_gpu = cuda.to_device(biases_cpu)
            
            self.weights_gpu.append(weights_gpu)
            self.biases_gpu.append(biases_gpu)
    
    def _allocate_gpu_memory(self):
        """Pre-allocate GPU memory for forward/backward passes"""
        max_batch_size = 4096  # Scaled for RTX 5090
        
        # Allocate memory for each layer's outputs
        for layer_size in self.layer_sizes:
            layer_output = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            layer_input = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            gradient = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            
            self.layer_outputs_gpu.append(layer_output)
            self.layer_inputs_gpu.append(layer_input)
            self.gradients_gpu.append(gradient)
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent memory leaks"""
        try:
            # Clear all GPU arrays
            self.layer_outputs_gpu.clear()
            self.layer_inputs_gpu.clear()
            self.gradients_gpu.clear()
            self.weights_gpu.clear()
            self.biases_gpu.clear()
            
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
    
    def forward(self, input_batch_gpu):
        """
        Forward pass entirely on GPU using RTX 5090 optimized kernels
        input_batch_gpu: GPU array of shape (batch_size, vector_length)
        returns: GPU array of shape (batch_size, vector_length)
        """
        # Ensure we're on the right device
        cuda.select_device(0)
        
        current_output = input_batch_gpu
        batch_size = input_batch_gpu.shape[0]
        
        # Store input for backward pass
        gpu_copy_array(current_output, self.layer_inputs_gpu[0][:batch_size])
        cuda.synchronize()  # Force GPU execution
        
        # Forward through all layers
        for i in range(len(self.weights_gpu)):
            # Linear transformation: output = input @ weights + bias
            # Use the correct output array - layer i outputs to layer_sizes[i+1] dimensions
            temp_output = self.layer_outputs_gpu[i+1][:batch_size]
            
            # Matrix multiplication
            launch_matrix_multiply(current_output, self.weights_gpu[i], temp_output)
            cuda.synchronize()  # Force GPU execution
            
            # Add bias
            gpu_add_bias(temp_output, self.biases_gpu[i], temp_output)
            cuda.synchronize()  # Force GPU execution
            
            # Apply activation (except for last layer)
            if i < len(self.weights_gpu) - 1:  # Not output layer
                gpu_leaky_relu_forward(temp_output, temp_output, self.alpha)
                cuda.synchronize()  # Force GPU execution
            
            # Store for backward pass
            if i < len(self.layer_inputs_gpu) - 1:
                gpu_copy_array(temp_output, self.layer_inputs_gpu[i + 1][:batch_size])
                cuda.synchronize()  # Force GPU execution
            
            current_output = temp_output
        
        return current_output
    
    def backward(self, input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm):
        """
        Backward pass with RTX 5090 optimized kernels and proper memory bounds checking
        """
        batch_size = input_batch_gpu.shape[0]
        max_batch_size = self.layer_outputs_gpu[0].shape[0]
        
        # Safety check to prevent illegal memory access
        if batch_size > max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds pre-allocated memory {max_batch_size}")
        
        # Forward pass to get outputs (uses pre-allocated memory)
        final_output = self.forward(input_batch_gpu)
        
        # Compute initial gradient (MSE loss)
        current_grad = self.gradients_gpu[-1][:batch_size]
        gpu_mse_gradient(final_output, target_batch_gpu, current_grad)
        
        # Backward through all layers
        for i in reversed(range(len(self.weights_gpu))):
            layer_input = self.layer_inputs_gpu[i][:batch_size]
            
            # Compute weight gradients
            weight_grad = cuda.device_array_like(self.weights_gpu[i])
            # Use optimized transpose instead of .T operation
            layer_input_T = create_transpose_gpu(layer_input)
            launch_matrix_multiply(layer_input_T, current_grad, weight_grad)
            
            # Compute bias gradients (simplified - sum over batch dimension)
            bias_grad = cuda.device_array_like(self.biases_gpu[i])
            # Note: This is simplified - proper implementation would need a reduction kernel
            
            # Clip gradients
            gpu_gradient_clip(weight_grad, grad_clip_norm)
            gpu_gradient_clip(bias_grad, grad_clip_norm)
            
            # Update weights
            gpu_update_weights(self.weights_gpu[i], weight_grad, learning_rate)
            gpu_update_weights(self.biases_gpu[i], bias_grad, learning_rate)
            
            # Compute gradient for previous layer
            if i > 0:
                prev_grad = self.gradients_gpu[i - 1][:batch_size]
                # Use optimized transpose instead of .T operation
                weights_T = create_transpose_gpu(self.weights_gpu[i])
                launch_matrix_multiply(current_grad, weights_T, prev_grad)
                
                # Apply activation derivative
                layer_input_prev = self.layer_inputs_gpu[i - 1][:batch_size]
                gpu_leaky_relu_backward(layer_input_prev, prev_grad, prev_grad, self.alpha)
                
                current_grad = prev_grad
        
        # Force GPU execution
        cuda.synchronize()
    
    def train_batch(self, batch_data_cpu, learning_rate, grad_clip_norm):
        """
        Train on a batch with minimal CPU-GPU transfers and proper memory bounds checking
        Only transfers the small batch data, keeps weights on GPU
        """
        # Check if batch size exceeds pre-allocated memory
        batch_size = batch_data_cpu.shape[0]
        max_batch_size = self.layer_outputs_gpu[0].shape[0]  # Get pre-allocated size
        
        if batch_size > max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds pre-allocated memory {max_batch_size}")
        
        # Transfer only the small batch to GPU (much smaller than full dataset)
        input_batch_gpu = cuda.to_device(batch_data_cpu.astype(np.float32))
        target_batch_gpu = input_batch_gpu  # Autoencoder targets = inputs
        
        # Forward and backward pass with proper bounds checking
        self.backward(input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm)
        
        # Compute loss using GPU kernel
        final_output = self.forward(input_batch_gpu)
        
        # GPU-based MSE loss calculation
        loss_gpu = cuda.device_array_like(final_output)
        gpu_mse_loss(final_output, target_batch_gpu, loss_gpu)
        
        # Return scalar loss (minimal GPU-CPU transfer)
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
        
        for w_gpu, b_gpu in zip(self.weights_gpu, self.biases_gpu):
            weights_cpu.append(w_gpu.copy_to_host())
            biases_cpu.append(b_gpu.copy_to_host())
        
        return weights_cpu, biases_cpu
    
    def load_weights(self, weights_cpu, biases_cpu):
        """
        Load weights from CPU arrays
        """
        for i, (w_cpu, b_cpu) in enumerate(zip(weights_cpu, biases_cpu)):
            self.weights_gpu[i] = cuda.to_device(w_cpu.astype(np.float32))
            self.biases_gpu[i] = cuda.to_device(b_cpu.astype(np.float32))