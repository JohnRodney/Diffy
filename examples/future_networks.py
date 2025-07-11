#!/usr/bin/env python3
"""
Examples of how to extend the base network architecture for new modalities

This demonstrates the scalability of the new architecture - you can add
image encoders, audio encoders, multimodal networks, etc. while keeping
all the same analysis utilities working.
"""

import numpy as np
from typing import Dict, Any
from neuralnet.base_network import BaseNetwork, NetworkFactory
from neuralnet.base_layer import create_layer

class ImageAutoencoder(BaseNetwork):
    """Example: Image autoencoder that could work with your analysis tools"""
    
    def __init__(self, config: Dict[str, Any]):
        self.image_height = config['image_height']
        self.image_width = config['image_width'] 
        self.channels = config['channels']
        self.latent_dim = config['latent_dim']
        
        super().__init__(config)
    
    def _build_network(self):
        """Build a simple image autoencoder"""
        input_size = self.image_height * self.image_width * self.channels
        
        # Encoder: progressively reduce dimensions
        self.encoder_layers = [
            create_layer('dense', input_size, 512, activation='relu'),
            create_layer('dense', 512, 256, activation='relu'),
            create_layer('dense', 256, 128, activation='relu'),
            create_layer('dense', 128, self.latent_dim, activation='linear')
        ]
        
        # Decoder: progressively expand dimensions  
        self.decoder_layers = [
            create_layer('dense', self.latent_dim, 128, activation='relu'),
            create_layer('dense', 128, 256, activation='relu'),
            create_layer('dense', 256, 512, activation='relu'),
            create_layer('dense', 512, input_size, activation='sigmoid', is_output_layer=True)
        ]
        
        self.all_layers = self.encoder_layers + self.decoder_layers
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through image autoencoder"""
        # Flatten images if needed
        if len(input_data.shape) > 2:
            batch_size = input_data.shape[0]
            input_data = input_data.reshape(batch_size, -1)
        
        current_output = input_data
        for layer in self.all_layers:
            current_output = layer.forward(current_output)
        
        return current_output
    
    def backward(self, target_batch: np.ndarray, final_output: np.ndarray, 
                learning_rate: float, grad_clip_norm: float) -> np.ndarray:
        """Backward pass - same pattern as text autoencoder"""
        # Implementation would be similar to TextAutoencoder
        return np.zeros_like(target_batch)  # Placeholder return

class MultimodalEncoder(BaseNetwork):
    """Example: Network that handles both text and images"""
    
    def __init__(self, config: Dict[str, Any]):
        self.text_dim = config['text_dim']
        self.image_dim = config['image_dim'] 
        self.shared_latent_dim = config['shared_latent_dim']
        
        super().__init__(config)
    
    def _build_network(self):
        """Build separate encoders that project to shared space"""
        # Text encoder branch
        self.text_encoder = [
            create_layer('dense', self.text_dim, 256, activation='relu'),
            create_layer('dense', 256, 128, activation='relu'),
            create_layer('dense', 128, self.shared_latent_dim, activation='linear')
        ]
        
        # Image encoder branch  
        self.image_encoder = [
            create_layer('dense', self.image_dim, 512, activation='relu'),
            create_layer('dense', 512, 256, activation='relu'),
            create_layer('dense', 256, self.shared_latent_dim, activation='linear')
        ]
        
        # Shared decoder
        self.shared_decoder = [
            create_layer('dense', self.shared_latent_dim, 256, activation='relu'),
            create_layer('dense', 256, self.text_dim + self.image_dim, is_output_layer=True)
        ]
        
        self.all_layers = self.text_encoder + self.image_encoder + self.shared_decoder
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through multimodal network"""
        # Split input into text and image parts
        text_data = input_data[:, :self.text_dim]
        image_data = input_data[:, self.text_dim:]
        
        # Encode both modalities
        text_latent = text_data
        for layer in self.text_encoder:
            text_latent = layer.forward(text_latent)
        
        image_latent = image_data  
        for layer in self.image_encoder:
            image_latent = layer.forward(image_latent)
        
        # Combine in shared space (could be average, concat, attention, etc.)
        shared_latent = (text_latent + image_latent) / 2
        
        # Decode to reconstruction
        output = shared_latent
        for layer in self.shared_decoder:
            output = layer.forward(output)
        
        return output
    
    def backward(self, target_batch: np.ndarray, final_output: np.ndarray, 
                learning_rate: float, grad_clip_norm: float) -> np.ndarray:
        """Backward pass for multimodal network"""
        # Implementation would handle gradients flowing back through both branches
        return np.zeros_like(target_batch)  # Placeholder return

# Register the new network types
NetworkFactory.register('image_autoencoder', ImageAutoencoder)
NetworkFactory.register('multimodal_encoder', MultimodalEncoder)

def example_usage():
    """Show how to use the new architectures"""
    
    # Create an image autoencoder
    image_config = {
        'image_height': 32,
        'image_width': 32, 
        'channels': 3,
        'latent_dim': 64
    }
    image_net = NetworkFactory.create('image_autoencoder', image_config)
    
    # Create a multimodal network
    multimodal_config = {
        'text_dim': 128,
        'image_dim': 1024,
        'shared_latent_dim': 256  
    }
    multimodal_net = NetworkFactory.create('multimodal_encoder', multimodal_config)
    
    # ALL your existing analysis utilities work with these new networks!
    from utils.drift_analysis import calculate_vector_drift
    from utils.checkpoint_manager import save_checkpoint
    from utils.training_analysis import analyze_training_progression
    
    print("✅ New network types created successfully!")
    print(f"📊 Image autoencoder: {image_net.get_architecture_info()['total_parameters']} parameters")
    print(f"🔗 Multimodal encoder: {multimodal_net.get_architecture_info()['total_parameters']} parameters")
    print("\n🎯 Key Benefits:")
    print("• Same analysis utilities work for all network types")
    print("• Same checkpoint/drift tracking for any architecture") 
    print("• Easy to add new modalities without breaking existing code")
    print("• Pluggable activation functions and layer types")

if __name__ == "__main__":
    example_usage() 