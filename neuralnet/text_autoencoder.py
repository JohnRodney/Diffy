#!/usr/bin/env python3
"""
Text autoencoder network that extends the base network class
"""

import numpy as np
import math
from typing import Dict, Any
from .base_network import BaseNetwork, NetworkFactory
from .base_layer import create_layer

class TextAutoencoder(BaseNetwork):
    """Text autoencoder for memorization experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        # Extract autoencoder-specific config
        self.vector_length = config['vector_length']
        self.hidden_layer_count = config['hidden_layer_count']
        self.bottleneck_size = config['bottleneck_size']
        self.activation_type = config.get('activation_type', 'leaky_relu')
        self.activation_alpha = config.get('activation_alpha', 0.01)
        
        # Call parent constructor
        super().__init__(config)
    
    def _build_network(self):
        """Build the text autoencoder architecture"""
        current_hidden_layer_size = self.vector_length
        reduction_factor = math.ceil((self.vector_length - self.bottleneck_size) / self.hidden_layer_count)
        
        # Encoder Layers
        self.encoder_layers = []
        
        # Input layer
        self.encoder_layers.append(
            create_layer('dense', self.vector_length, current_hidden_layer_size, 
                        activation=self.activation_type, alpha=self.activation_alpha)
        )
        
        # Hidden encoder layers
        for _ in range(self.hidden_layer_count):
            last_hidden_layer_size = current_hidden_layer_size
            next_hidden_layer_size = current_hidden_layer_size - reduction_factor
            if next_hidden_layer_size < self.bottleneck_size:
                next_hidden_layer_size = self.bottleneck_size
            current_hidden_layer_size = next_hidden_layer_size
            
            self.encoder_layers.append(
                create_layer('dense', last_hidden_layer_size, current_hidden_layer_size,
                            activation=self.activation_type, alpha=self.activation_alpha)
            )
        
        # Decoder Layers  
        self.decoder_layers = []
        
        # Bottleneck layer (input to decoder)
        self.decoder_layers.append(
            create_layer('dense', current_hidden_layer_size, current_hidden_layer_size,
                        activation=self.activation_type, alpha=self.activation_alpha)
        )
        
        # Hidden decoder layers
        for _ in range(self.hidden_layer_count):
            prev_layer_size = current_hidden_layer_size
            next_hidden_layer_size = current_hidden_layer_size + reduction_factor
            if next_hidden_layer_size > self.vector_length:
                next_hidden_layer_size = self.vector_length
            current_hidden_layer_size = next_hidden_layer_size
            
            self.decoder_layers.append(
                create_layer('dense', prev_layer_size, current_hidden_layer_size,
                            activation=self.activation_type, alpha=self.activation_alpha)
            )
        
        # Output layer (linear activation)
        self.decoder_output_layer = create_layer('dense', current_hidden_layer_size, self.vector_length, 
                                                 is_output_layer=True)
        self.decoder_layers.append(self.decoder_output_layer)
        
        # Combine all layers for easy access
        self.all_layers = self.encoder_layers + self.decoder_layers
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the entire autoencoder"""
        current_output = input_data
        
        for layer in self.all_layers:
            current_output = layer.forward(current_output)
            if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                return np.full_like(current_output, np.nan)  # Propagate NaN/Inf
        
        return current_output
    
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Encode input to bottleneck representation"""
        current_output = input_data
        
        for layer in self.encoder_layers:
            current_output = layer.forward(current_output)
            if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                return np.full_like(current_output, np.nan)
        
        return current_output
    
    def decode(self, latent_data: np.ndarray) -> np.ndarray:
        """Decode latent representation to output"""
        current_output = latent_data
        
        for layer in self.decoder_layers:
            current_output = layer.forward(current_output)
            if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                return np.full_like(current_output, np.nan)
        
        return current_output
    
    def backward(self, target_batch: np.ndarray, final_reconstruction: np.ndarray, 
                learning_rate: float, grad_clip_norm: float) -> np.ndarray:
        """Backward pass through the autoencoder"""
        # Start backward pass from the output layer
        incoming_gradient = self.decoder_output_layer.backward_output_layer(
            target_batch, final_reconstruction, learning_rate, grad_clip_norm
        )
        
        if np.any(np.isnan(incoming_gradient)) or np.any(np.isinf(incoming_gradient)):
            return np.full_like(incoming_gradient, np.nan)  # Indicate NaN/Inf in backward pass
        
        # Propagate through remaining layers in reverse
        # We iterate through all_layers[:-1] because decoder_output_layer was handled separately
        for layer in reversed(self.all_layers[:-1]):
            incoming_gradient = layer.backward(incoming_gradient, learning_rate, grad_clip_norm)
            if np.any(np.isnan(incoming_gradient)) or np.any(np.isinf(incoming_gradient)):
                return np.full_like(incoming_gradient, np.nan)  # Indicate NaN/Inf in backward pass
        
        return incoming_gradient  # This return value is typically discarded after the first layer
    
    def get_bottleneck_info(self) -> Dict[str, Any]:
        """Get information specific to the autoencoder bottleneck"""
        bottleneck_layer = self.encoder_layers[-1]  # Last encoder layer
        
        return {
            'bottleneck_size': self.bottleneck_size,
            'compression_ratio': self.vector_length / self.bottleneck_size,
            'dimensions_per_item': self.bottleneck_size / self.config.get('total_items', 1),
            'bottleneck_layer_index': len(self.encoder_layers) - 1,
            'bottleneck_activation': bottleneck_layer.activation.name
        }
    
    def test_reconstruction(self, input_vector: np.ndarray, tokenizer) -> Dict[str, Any]:
        """Test reconstruction quality for a single input"""
        reconstructed = self.forward(input_vector)
        
        if np.any(np.isnan(reconstructed)) or np.any(np.isinf(reconstructed)):
            return {
                'reconstruction_error': np.nan,
                'cosine_similarity': -2.0,
                'best_match': "NaN/Inf",
                'is_successful': False
            }
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((input_vector - reconstructed) ** 2)
        
        # Calculate cosine similarity
        dot_product = np.dot(input_vector.flatten(), reconstructed.flatten())
        input_norm = np.linalg.norm(input_vector.flatten())
        reconstructed_norm = np.linalg.norm(reconstructed.flatten())
        
        if input_norm > 0 and reconstructed_norm > 0:
            cosine_similarity = dot_product / (input_norm * reconstructed_norm)
        else:
            cosine_similarity = -1.0
        
        return {
            'reconstruction_error': float(reconstruction_error),
            'cosine_similarity': float(cosine_similarity),
            'reconstructed_vector': reconstructed,
            'is_successful': cosine_similarity > 0.8  # Threshold for "good" reconstruction
        }

# Register the text autoencoder with the factory
NetworkFactory.register('text_autoencoder', TextAutoencoder)

# Backward compatibility - create a simple function that matches the old interface
def create_text_autoencoder(vector_length: int, hidden_layer_count: int, 
                           bottleneck_size: int, activation_alpha: float = 0.01) -> TextAutoencoder:
    """Create a text autoencoder with the old-style parameters"""
    config = {
        'vector_length': vector_length,
        'hidden_layer_count': hidden_layer_count,
        'bottleneck_size': bottleneck_size,
        'activation_type': 'leaky_relu',
        'activation_alpha': activation_alpha
    }
    
    return TextAutoencoder(config) 