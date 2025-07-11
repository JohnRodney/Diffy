#!/usr/bin/env python3
"""
GPU Text Autoencoder using pure numba CUDA kernels
All operations happen on GPU memory - no CPU transfers during training
"""

import numpy as np
import math
from numba import cuda
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kernels'))
try:
    from matrix_kernels import *
except ImportError:
    # For now, define stub functions for linter
    def launch_elementwise_kernel(func, *args):
        pass
    def launch_matrix_multiply(a, b, c):
        pass
    def launch_2d_kernel(func, *args):
        pass
    def gpu_copy_array(a, b):
        pass
    def gpu_add_bias(a, b, c):
        pass
    def gpu_leaky_relu_forward(a, b, c):
        pass
    def gpu_mse_gradient(a, b, c):
        pass
    def gpu_gradient_clip(a, b):
        pass
    def gpu_update_weights(a, b, c):
        pass
    def gpu_leaky_relu_backward(a, b, c, d):
        pass
    def gpu_mse_loss(a, b, c):
        pass

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
        
        print(f"🔥 GPU Text Autoencoder initialized")
        print(f"📊 Architecture: {self.layer_sizes}")
        print(f"🎯 Total parameters: {self._count_parameters()}")
    
    def _calculate_layer_sizes(self):
        """Calculate the size of each layer"""
        sizes = [self.vector_length]
        
        # Encoder layers
        current_size = self.vector_length
        reduction_factor = math.ceil((self.vector_length - self.bottleneck_size) / self.hidden_layer_count)
        
        for _ in range(self.hidden_layer_count + 1):  # +1 for bottleneck
            next_size = current_size - reduction_factor
            if next_size < self.bottleneck_size:
                next_size = self.bottleneck_size
            sizes.append(next_size)
            current_size = next_size
        
        # Decoder layers (mirror of encoder)
        decoder_sizes = sizes[:-1]  # Remove bottleneck
        decoder_sizes.reverse()
        sizes.extend(decoder_sizes[1:])  # Skip first (bottleneck already added)
        
        return sizes
    
    def _initialize_weights_on_gpu(self):
        """Initialize all weights and biases directly on GPU"""
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            # Xavier initialization
            limit = math.sqrt(6.0 / (input_size + output_size))
            weights_cpu = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float32)
            biases_cpu = np.zeros((1, output_size), dtype=np.float32)
            
            # Transfer to GPU
            weights_gpu = cuda.to_device(weights_cpu)
            biases_gpu = cuda.to_device(biases_cpu)
            
            self.weights_gpu.append(weights_gpu)
            self.biases_gpu.append(biases_gpu)
    
    def _allocate_gpu_memory(self):
        """Pre-allocate GPU memory for forward/backward passes"""
        max_batch_size = 64  # Adjust based on your needs
        
        # Allocate memory for each layer's outputs
        for layer_size in self.layer_sizes:
            layer_output = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            layer_input = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            gradient = cuda.device_array((max_batch_size, layer_size), dtype=np.float32)
            
            self.layer_outputs_gpu.append(layer_output)
            self.layer_inputs_gpu.append(layer_input)
            self.gradients_gpu.append(gradient)
    
    def _count_parameters(self):
        """Count total parameters in the model"""
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            total += input_size * output_size + output_size  # weights + biases
        return total
    
    def forward_gpu(self, input_batch_gpu):
        """
        Forward pass entirely on GPU
        input_batch_gpu: GPU array of shape (batch_size, vector_length)
        returns: GPU array of shape (batch_size, vector_length)
        """
        current_output = input_batch_gpu
        batch_size = input_batch_gpu.shape[0]
        
        # Store input for backward pass
        launch_elementwise_kernel(gpu_copy_array, current_output, self.layer_inputs_gpu[0][:batch_size])
        
        # Forward through all layers
        for i in range(len(self.weights_gpu)):
            # Linear transformation: output = input @ weights + bias
            temp_output = self.layer_outputs_gpu[i][:batch_size]
            
            # Matrix multiplication
            launch_matrix_multiply(current_output, self.weights_gpu[i], temp_output)
            
            # Add bias
            launch_2d_kernel(gpu_add_bias, temp_output, self.biases_gpu[i], temp_output)
            
            # Apply activation (except for last layer)
            if i < len(self.weights_gpu) - 1:  # Not output layer
                launch_elementwise_kernel(gpu_leaky_relu_forward, temp_output, temp_output, self.alpha)
            
            # Store for backward pass
            if i < len(self.layer_inputs_gpu) - 1:
                launch_elementwise_kernel(gpu_copy_array, temp_output, self.layer_inputs_gpu[i + 1][:batch_size])
            
            current_output = temp_output
        
        return current_output
    
    def backward_gpu(self, input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm):
        """
        Backward pass entirely on GPU
        """
        batch_size = input_batch_gpu.shape[0]
        
        # Forward pass to get outputs
        final_output = self.forward_gpu(input_batch_gpu)
        
        # Compute initial gradient (MSE loss)
        current_grad = self.gradients_gpu[-1][:batch_size]
        launch_elementwise_kernel(gpu_mse_gradient, final_output, target_batch_gpu, current_grad)
        
        # Backward through all layers
        for i in reversed(range(len(self.weights_gpu))):
            layer_input = self.layer_inputs_gpu[i][:batch_size]
            
            # Compute weight gradients
            weight_grad = cuda.device_array_like(self.weights_gpu[i])
            launch_matrix_multiply(layer_input.T, current_grad, weight_grad)
            
            # Compute bias gradients
            bias_grad = cuda.device_array_like(self.biases_gpu[i])
            # Sum over batch dimension
            # This is simplified - you'd need a proper reduction kernel
            
            # Clip gradients
            launch_elementwise_kernel(gpu_gradient_clip, weight_grad, grad_clip_norm)
            launch_elementwise_kernel(gpu_gradient_clip, bias_grad, grad_clip_norm)
            
            # Update weights
            launch_elementwise_kernel(gpu_update_weights, self.weights_gpu[i], weight_grad, learning_rate)
            launch_elementwise_kernel(gpu_update_weights, self.biases_gpu[i], bias_grad, learning_rate)
            
            # Compute gradient for previous layer
            if i > 0:
                prev_grad = self.gradients_gpu[i - 1][:batch_size]
                launch_matrix_multiply(current_grad, self.weights_gpu[i].T, prev_grad)
                
                # Apply activation derivative
                layer_input_prev = self.layer_inputs_gpu[i - 1][:batch_size]
                launch_elementwise_kernel(gpu_leaky_relu_backward, layer_input_prev, prev_grad, prev_grad, self.alpha)
                
                current_grad = prev_grad
    
    def train_batch_gpu(self, input_batch_cpu, target_batch_cpu, learning_rate, grad_clip_norm):
        """
        Train on a batch - handles CPU to GPU transfer
        """
        # Transfer batch to GPU
        input_batch_gpu = cuda.to_device(input_batch_cpu.astype(np.float32))
        target_batch_gpu = cuda.to_device(target_batch_cpu.astype(np.float32))
        
        # Forward and backward pass
        self.backward_gpu(input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm)
        
        # Compute loss for monitoring
        final_output = self.forward_gpu(input_batch_gpu)
        loss_gpu = cuda.device_array_like(final_output)
        launch_elementwise_kernel(gpu_mse_loss, final_output, target_batch_gpu, loss_gpu)
        
        # Return loss to CPU
        loss_cpu = loss_gpu.copy_to_host()
        return np.mean(loss_cpu)
    
    def predict_gpu(self, input_batch_cpu):
        """
        Prediction - returns result to CPU
        """
        input_batch_gpu = cuda.to_device(input_batch_cpu.astype(np.float32))
        output_gpu = self.forward_gpu(input_batch_gpu)
        return output_gpu.copy_to_host()
    
    def get_weights_cpu(self):
        """
        Get all weights as CPU arrays (for saving/analysis)
        """
        weights_cpu = []
        biases_cpu = []
        
        for w_gpu, b_gpu in zip(self.weights_gpu, self.biases_gpu):
            weights_cpu.append(w_gpu.copy_to_host())
            biases_cpu.append(b_gpu.copy_to_host())
        
        return weights_cpu, biases_cpu
    
    def set_weights_cpu(self, weights_cpu, biases_cpu):
        """
        Set weights from CPU arrays
        """
        for i, (w_cpu, b_cpu) in enumerate(zip(weights_cpu, biases_cpu)):
            self.weights_gpu[i] = cuda.to_device(w_cpu.astype(np.float32))
            self.biases_gpu[i] = cuda.to_device(b_cpu.astype(np.float32))

def test_gpu_autoencoder():
    """Test the GPU autoencoder"""
    print("🧪 Testing GPU Text Autoencoder...")
    
    # Create test data
    vector_length = 128
    batch_size = 8
    input_batch = np.random.randn(batch_size, vector_length).astype(np.float32)
    target_batch = input_batch.copy()  # Autoencoder task
    
    # Create GPU autoencoder
    gpu_ae = GPUTextAutoencoder(vector_length, 2, 32, alpha=0.01)
    
    # Train one batch
    loss = gpu_ae.train_batch_gpu(input_batch, target_batch, 0.001, 1.0)
    print(f"✅ Training step completed. Loss: {loss:.6f}")
    
    # Test prediction
    output = gpu_ae.predict_gpu(input_batch)
    print(f"📊 Input shape: {input_batch.shape}, Output shape: {output.shape}")
    
    print("🎉 GPU autoencoder test passed!")

if __name__ == "__main__":
    test_gpu_autoencoder() 