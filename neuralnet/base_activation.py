#!/usr/bin/env python3
"""
Base activation function interface for neural network layers
"""

import numpy as np
from abc import ABC, abstractmethod

class BaseActivation(ABC):
    """Abstract base class for activation functions"""
    
    @abstractmethod
    def forward(self, x):
        """Apply activation function to input"""
        pass
    
    @abstractmethod
    def backward(self, x):
        """Compute derivative of activation function"""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return name of activation function"""
        pass

class ReLUActivation(BaseActivation):
    """ReLU activation function"""
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)
    
    @property
    def name(self):
        return "relu"

class LeakyReLUActivation(BaseActivation):
    """Leaky ReLU activation function with configurable alpha"""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return np.maximum(self.alpha * x, x)
    
    def backward(self, x):
        return np.where(x > 0, 1.0, self.alpha)
    
    @property
    def name(self):
        return f"leaky_relu_alpha_{self.alpha}"

class SigmoidActivation(BaseActivation):
    """Sigmoid activation function"""
    
    def forward(self, x):
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def backward(self, x):
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    @property
    def name(self):
        return "sigmoid"

class TanhActivation(BaseActivation):
    """Tanh activation function"""
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2
    
    @property
    def name(self):
        return "tanh"

class LinearActivation(BaseActivation):
    """Linear activation (identity function) for output layers"""
    
    def forward(self, x):
        return x
    
    def backward(self, x):
        return np.ones_like(x)
    
    @property
    def name(self):
        return "linear"

# Factory function for creating activation functions
def create_activation(activation_type, **kwargs):
    """Factory function to create activation functions"""
    activations = {
        'relu': ReLUActivation,
        'leaky_relu': LeakyReLUActivation,
        'sigmoid': SigmoidActivation,
        'tanh': TanhActivation,
        'linear': LinearActivation
    }
    
    if activation_type not in activations:
        raise ValueError(f"Unknown activation type: {activation_type}. Available: {list(activations.keys())}")
    
    return activations[activation_type](**kwargs) 