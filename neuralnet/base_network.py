#!/usr/bin/env python3
"""
Base network class that defines the interface for all neural network architectures
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseNetwork(ABC):
    """Abstract base class for all neural network architectures"""
    
    def __init__(self, network_config: Dict[str, Any]):
        self.config = network_config
        self.all_layers = []
        self.network_type = self.__class__.__name__
        
        # Build the network architecture
        self._build_network()
    
    @abstractmethod
    def _build_network(self):
        """Build the network architecture (implementation specific)"""
        pass
    
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        pass
    
    @abstractmethod
    def backward(self, target_batch: np.ndarray, final_output: np.ndarray, 
                learning_rate: float, grad_clip_norm: float) -> np.ndarray:
        """Backward pass through the network"""
        pass
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the network architecture"""
        layer_info = []
        for i, layer in enumerate(self.all_layers):
            layer_info.append({
                'layer_index': i,
                'layer_type': layer.__class__.__name__,
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'activation': layer.activation.name,
                'is_output_layer': layer.is_output_layer,
                'parameter_count': layer.weights.size + layer.biases.size
            })
        
        total_params = sum(info['parameter_count'] for info in layer_info)
        
        return {
            'network_type': self.network_type,
            'total_layers': len(self.all_layers),
            'total_parameters': total_params,
            'config': self.config,
            'layers': layer_info
        }
    
    def get_layer_outputs(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Get outputs from each layer (useful for analysis)"""
        outputs = []
        current_output = input_data
        
        for layer in self.all_layers:
            current_output = layer.forward(current_output)
            outputs.append(current_output.copy())
        
        return outputs
    
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Encode input to latent representation (for autoencoders)"""
        # Default implementation: run full forward pass
        return self.forward(input_data)
    
    def decode(self, latent_data: np.ndarray) -> np.ndarray:
        """Decode latent representation (for autoencoders)"""
        # Default implementation: not supported for all network types
        raise NotImplementedError(f"Decode not implemented for {self.network_type}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions (for classifiers/regressors)"""
        # Default implementation: run forward pass
        return self.forward(input_data)
    
    def save_weights(self) -> Dict[str, np.ndarray]:
        """Save all network weights and biases"""
        weights_dict = {}
        for i, layer in enumerate(self.all_layers):
            weights_dict[f"layer_{i}_weights"] = layer.weights
            weights_dict[f"layer_{i}_biases"] = layer.biases
        return weights_dict
    
    def load_weights(self, weights_dict: Dict[str, np.ndarray]) -> bool:
        """Load network weights and biases"""
        try:
            for i, layer in enumerate(self.all_layers):
                weight_key = f"layer_{i}_weights"
                bias_key = f"layer_{i}_biases"
                
                if weight_key in weights_dict and bias_key in weights_dict:
                    layer.weights = weights_dict[weight_key]
                    layer.biases = weights_dict[bias_key]
                else:
                    print(f"Warning: Parameters for layer {i} not found in weights dict.")
                    return False
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def validate_input(self, input_data: np.ndarray) -> bool:
        """Validate input data shape and type"""
        if not isinstance(input_data, np.ndarray):
            return False
        
        if len(input_data.shape) != 2:
            return False
        
        if input_data.shape[1] != self.all_layers[0].input_size:
            return False
        
        return True
    
    def get_gradient_norms(self) -> Dict[str, float]:
        """Get gradient norms for each layer (for monitoring training)"""
        # This would be implemented during training
        # For now, return empty dict
        return {}

class NetworkFactory:
    """Factory class for creating different network types"""
    
    _network_registry = {}
    
    @classmethod
    def register(cls, network_type: str, network_class):
        """Register a network class with a type name"""
        cls._network_registry[network_type] = network_class
    
    @classmethod
    def create(cls, network_type: str, config: Dict[str, Any]) -> BaseNetwork:
        """Create a network of the specified type"""
        if network_type not in cls._network_registry:
            raise ValueError(f"Unknown network type: {network_type}. "
                           f"Available types: {list(cls._network_registry.keys())}")
        
        network_class = cls._network_registry[network_type]
        return network_class(config)
    
    @classmethod
    def list_available_types(cls) -> List[str]:
        """List all available network types"""
        return list(cls._network_registry.keys()) 