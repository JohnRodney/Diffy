#!/usr/bin/env python3
"""
Base layer class for neural network layers with pluggable activation functions
"""

import numpy as np
from abc import ABC, abstractmethod
from .base_activation import create_activation, LinearActivation

class BaseLayer(ABC):
    """Abstract base class for neural network layers"""
    
    def __init__(self, input_size, output_size, activation='relu', is_output_layer=False, **activation_kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.is_output_layer = is_output_layer
        
        # Set activation function
        if is_output_layer:
            self.activation = LinearActivation()
        else:
            self.activation = create_activation(activation, **activation_kwargs)
        
        # Initialize weights and biases
        self._initialize_parameters()
        
        # Store last inputs for backward pass
        self.last_input = None
        self.last_pre_activation = None
    
    @abstractmethod
    def _initialize_parameters(self):
        """Initialize weights and biases (implementation specific)"""
        pass
    
    def forward(self, input_data):
        """Forward pass through the layer"""
        self.last_input = input_data.copy()
        
        # Linear transformation
        self.last_pre_activation = np.dot(input_data, self.weights) + self.biases
        
        # Apply activation function
        if self.is_output_layer:
            # Output layer typically uses linear activation
            output = self.last_pre_activation
        else:
            output = self.activation.forward(self.last_pre_activation)
        
        return output
    
    @abstractmethod
    def backward(self, incoming_gradient, learning_rate, grad_clip_norm):
        """Backward pass through the layer"""
        pass
    
    def backward_output_layer(self, target_batch, final_reconstruction, learning_rate, grad_clip_norm):
        """Specialized backward pass for output layers"""
        batch_size = target_batch.shape[0]
        
        # Compute loss gradient (MSE)
        output_gradient = 2.0 * (final_reconstruction - target_batch) / batch_size
        
        # Since output layer uses linear activation, derivative is 1
        pre_activation_gradient = output_gradient
        
        # Compute gradients for weights and biases
        if self.last_input is not None:
            weight_gradient = np.dot(self.last_input.T, pre_activation_gradient)
        else:
            weight_gradient = np.zeros_like(self.weights)
        bias_gradient = np.sum(pre_activation_gradient, axis=0)
        
        # Apply gradient clipping
        weight_gradient = np.clip(weight_gradient, -grad_clip_norm, grad_clip_norm)
        bias_gradient = np.clip(bias_gradient, -grad_clip_norm, grad_clip_norm)
        
        # Update parameters
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        
        # Compute gradient to pass to previous layer
        incoming_gradient = np.dot(pre_activation_gradient, self.weights.T)
        
        return incoming_gradient
    
    def get_config(self):
        """Get layer configuration for serialization"""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation.name,
            'is_output_layer': self.is_output_layer
        }

class DenseLayer(BaseLayer):
    """Standard fully connected (dense) layer"""
    
    def _initialize_parameters(self):
        """Xavier/Glorot initialization for dense layers"""
        limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        self.biases = np.zeros((1, self.output_size))
    
    def backward(self, incoming_gradient, learning_rate, grad_clip_norm):
        """Backward pass for dense layer"""
        # Compute activation gradient
        if not self.is_output_layer:
            activation_gradient = self.activation.backward(self.last_pre_activation)
            pre_activation_gradient = incoming_gradient * activation_gradient
        else:
            pre_activation_gradient = incoming_gradient
        
        # Compute parameter gradients
        if self.last_input is not None:
            weight_gradient = np.dot(self.last_input.T, pre_activation_gradient)
        else:
            weight_gradient = np.zeros_like(self.weights)
        bias_gradient = np.sum(pre_activation_gradient, axis=0, keepdims=True)
        
        # Apply gradient clipping
        weight_gradient = np.clip(weight_gradient, -grad_clip_norm, grad_clip_norm)
        bias_gradient = np.clip(bias_gradient, -grad_clip_norm, grad_clip_norm)
        
        # Update parameters
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        
        # Compute gradient for previous layer
        incoming_gradient = np.dot(pre_activation_gradient, self.weights.T)
        
        return incoming_gradient

# Factory function for creating layers
def create_layer(layer_type, input_size, output_size, **kwargs):
    """Factory function to create different layer types"""
    layers = {
        'dense': DenseLayer,
        # Future layer types can be added here
        # 'conv2d': Conv2DLayer,
        # 'lstm': LSTMLayer,
        # etc.
    }
    
    if layer_type not in layers:
        raise ValueError(f"Unknown layer type: {layer_type}. Available: {list(layers.keys())}")
    
    return layers[layer_type](input_size, output_size, **kwargs) 