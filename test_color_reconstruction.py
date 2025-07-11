#!/usr/bin/env python3
"""
Advanced color reconstruction test
Actually runs inference on the models to see if they can reconstruct colors
"""

import numpy as np
import json
import os

class SimpleAutoencoder:
    """Simple autoencoder inference using loaded weights"""
    
    def __init__(self, model_path):
        self.model_data = np.load(model_path)
        self.weights = []
        self.biases = []
        
        # Extract weights and biases in order
        weight_keys = [k for k in self.model_data.keys() if 'weight' in k.lower()]
        bias_keys = [k for k in self.model_data.keys() if 'bias' in k.lower()]
        
        weight_keys.sort()
        bias_keys.sort()
        
        for w_key, b_key in zip(weight_keys, bias_keys):
            self.weights.append(self.model_data[w_key])
            self.biases.append(self.model_data[b_key])
        
        self.loss = self.model_data.get('loss', 'unknown')
        self.epoch = self.model_data.get('epoch', 'unknown')
        
        print(f"📊 Model loaded: {len(self.weights)} layers")
        print(f"   Loss: {self.loss}")
        print(f"   Epoch: {self.epoch}")
    
    def leaky_relu(self, x, alpha=0.01):
        """LeakyReLU activation"""
        return np.maximum(alpha * x, x)
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        current = x
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            current = np.dot(current, weight) + bias
            
            # Apply LeakyReLU (except possibly the last layer)
            if i < len(self.weights) - 1:  # Not the last layer
                current = self.leaky_relu(current)
            else:
                # Last layer - might use different activation or none
                current = self.leaky_relu(current)
        
        return current

def load_sample_colors():
    """Load a few sample colors for testing"""
    try:
        with open('colors_a_f.json', 'r') as f:
            colors = json.load(f)
        return colors[:5]  # Just test first 5
    except:
        return [
            {"Name": "Red", "Hex(RGB)": "#FF0000", "Red(RGB)": "100%", "Green(RGB)": "0%", "Blue(RGB)": "0%"},
            {"Name": "Blue", "Hex(RGB)": "#0000FF", "Red(RGB)": "0%", "Green(RGB)": "0%", "Blue(RGB)": "100%"},
        ]

def color_to_vector(color_data, vector_length=1024):
    """Convert color to input vector (simplified version)"""
    features = []
    
    # RGB values (normalized)
    try:
        rgb = [
            float(color_data['Red(RGB)'].replace('%', '')) / 100.0,
            float(color_data['Green(RGB)'].replace('%', '')) / 100.0,
            float(color_data['Blue(RGB)'].replace('%', '')) / 100.0
        ]
        features.extend(rgb)
    except:
        features.extend([0.0, 0.0, 0.0])
    
    # Simple name encoding (first few characters as ASCII)
    name = color_data['Name'].lower()
    name_features = []
    for char in name[:10]:  # First 10 characters
        name_features.append(ord(char) / 255.0)  # Normalize ASCII
    
    while len(name_features) < 10:
        name_features.append(0.0)
    
    features.extend(name_features)
    
    # Pad to vector length
    while len(features) < vector_length:
        features.append(0.0)
    
    return np.array(features[:vector_length], dtype=np.float32)

def test_reconstruction(model_path):
    """Test color reconstruction with a model"""
    print(f"\n🧪 Testing reconstruction with {os.path.basename(model_path)}")
    
    # Load model
    model = SimpleAutoencoder(model_path)
    
    # Load test colors
    colors = load_sample_colors()
    
    print(f"🎨 Testing {len(colors)} colors...")
    
    for color in colors:
        # Convert to vector
        input_vector = color_to_vector(color)
        
        # Run through model
        reconstructed = model.forward(input_vector)
        
        # Compare input vs output
        mse = np.mean((input_vector - reconstructed) ** 2)
        
        # Extract RGB from input and output
        input_rgb = input_vector[:3]
        output_rgb = reconstructed[:3]
        
        print(f"  {color['Name'][:20]:20s}")
        print(f"    Input RGB:  [{input_rgb[0]:.3f}, {input_rgb[1]:.3f}, {input_rgb[2]:.3f}]")
        print(f"    Output RGB: [{output_rgb[0]:.3f}, {output_rgb[1]:.3f}, {output_rgb[2]:.3f}]")
        print(f"    RGB Error:  [{abs(input_rgb[0]-output_rgb[0]):.3f}, {abs(input_rgb[1]-output_rgb[1]):.3f}, {abs(input_rgb[2]-output_rgb[2]):.3f}]")
        print(f"    MSE: {mse:.6f}")
        
        # Check if reconstruction is reasonable
        rgb_error = np.mean(np.abs(input_rgb - output_rgb))
        if rgb_error < 0.1:
            print(f"    ✅ Good reconstruction (RGB error: {rgb_error:.3f})")
        elif rgb_error < 0.3:
            print(f"    ⚠️  Moderate reconstruction (RGB error: {rgb_error:.3f})")
        else:
            print(f"    ❌ Poor reconstruction (RGB error: {rgb_error:.3f})")
        
        print()

def main():
    """Main reconstruction test"""
    print("🔬 Advanced Color Reconstruction Test")
    print("=" * 50)
    
    # Test latest models
    models_dir = "downloaded_models"
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.npz')])
    
    # Test last 2 models
    for model_file in model_files[-2:]:
        model_path = os.path.join(models_dir, model_file)
        test_reconstruction(model_path)

if __name__ == "__main__":
    main() 