#!/usr/bin/env python3
"""
Simple smoke test for GPU autoencoder
"""

import numpy as np
import os
import argparse
import time
from gpu_networks.gpu_text_autoencoder import GPUTextAutoencoder
from color_loader import load_colors_from_json
from tokenizer.text_tokenizer import Tokenizer
from numba import cuda

def load_data(vector_length, vector_method="random"):
    """Load color data for training"""
    try:
        colors_data = load_colors_from_json("colors_a_f.json")
        if not colors_data:
            raise ValueError("No color data found")
    except FileNotFoundError:
        raise ValueError("colors_a_f.json not found")
    
    print(f"Loaded {len(colors_data)} colors")
    color_names = colors_data  # colors_data is already a list of color name strings
    
    if not color_names:
        raise ValueError("No color names found")
    
    training_data = []
    
    if vector_method == "tokenized":
        # Use tokenizer to create sparse vectors (the "bug" for research)
        tokenizer = Tokenizer(vocab_size=len(color_names))
        tokenizer.fill_dictionary(color_names)
        
        for name in color_names:
            tokens = tokenizer.tokenize(name)
            encoded = np.zeros(vector_length, dtype=np.float32)
            
            # Spread each token value across the vector with a sine wave pattern
            for token in tokens:
                token_value = float(token)
                # Create a sine wave pattern based on the token value
                for i in range(vector_length):
                    # Sine wave with frequency based on token value
                    wave_value = token_value * np.sin(2 * np.pi * i * token_value / vector_length)
                    encoded[i] += wave_value
            
            training_data.append((encoded, name))
    else:
        # Create random vectors for each color name (like CPU version)
        for name in color_names:
            # Generate unique random vector for this color
            encoded = np.random.rand(vector_length).astype(np.float32)
            training_data.append((encoded, name))
    
    return training_data

def save_model(gpu_ae, training_data_array, training_data, best_loss, best_loss_epoch, final_loss, total_epochs, params, vector_method, start_time, epoch, is_best=False):
    print("Saving real model outputs for analysis...")
    real_model_outputs = gpu_ae.infer(training_data_array)
    
    os.makedirs("/models", exist_ok=True)
    weights_cpu, biases_cpu = gpu_ae.export_weights()

    save_dict = {}
    for i, (w, b) in enumerate(zip(weights_cpu, biases_cpu)):
        save_dict[f'weight_{i}'] = w
        save_dict[f'bias_{i}'] = b
    
    save_dict['training_vectors'] = training_data_array
    save_dict['training_names'] = [item[1] for item in training_data]
    save_dict['model_outputs'] = real_model_outputs
    save_dict['best_loss'] = best_loss
    save_dict['best_loss_epoch'] = best_loss_epoch
    save_dict['final_loss'] = final_loss
    save_dict['final_epoch'] = total_epochs - 1
    save_dict['parameters'] = params
    save_dict['vector_method'] = vector_method
    save_dict['training_start_time'] = start_time
    end_time = time.time()
    training_duration = end_time - start_time
    save_dict['training_end_time'] = end_time
    save_dict['training_duration_seconds'] = training_duration
    save_dict['epochs_per_second'] = total_epochs / training_duration
    save_dict['dataset_size'] = len(training_data_array)
    
    filename = f"home_e{epoch}.npz"
    filepath = f"/app/models/{filename}{'_best' if is_best else ''}"
    
    np.savez_compressed(filepath, **save_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Autoencoder Parameter Sweep")
    parser.add_argument("--vector-method", default="random", choices=["random", "tokenized"], 
                        help="Method for generating input vectors")
    parser.add_argument("--bottleneck-size", type=int, default=128, 
                        help="Size of the bottleneck layer")
    parser.add_argument("--vector-length", type=int, default=512, 
                        help="Length of input/output vectors")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--max-home-distance", type=float, default=2.0, 
                        help="Maximum allowed distance from home position")
    parser.add_argument("--position-weight", type=float, default=1.0, 
                        help="Weight for home position penalty")
    
    args = parser.parse_args()
    
    def smoke_test_with_args(vector_method, bottleneck_size, vector_length, epochs, learning_rate, batch_size, max_home_distance, position_weight):
        
        params = {
            'vector_length': vector_length,
            'hidden_layer_count': 4,
            'bottleneck_size': bottleneck_size,
            'learning_rate': learning_rate,
            'leaky_relu_alpha': 0.01,
            'batch_size': batch_size,
            'grad_clip_norm': 1,
            "epochs": epochs,
            'max_home_distance': max_home_distance,
            'position_weight': position_weight,
        }
        
        print(f"Compression ratio: {params['vector_length'] / params['bottleneck_size']:.2f}x")
        print(f"Hidden layers: {params['hidden_layer_count']} (gradual geometric stepdown)")
        print(f"Home position: max_distance={max_home_distance}, weight={position_weight}")
        
        gpu_ae = GPUTextAutoencoder(
            vector_length=params['vector_length'],
            hidden_layer_count=params['hidden_layer_count'],
            bottleneck_size=params['bottleneck_size'],
            alpha=params['leaky_relu_alpha']
        )
            
        print(f"Layer architecture: {' → '.join(map(str, gpu_ae.layer_sizes))}")
            
        # Set home position parameters
        gpu_ae.set_home_position_params(max_home_distance, position_weight)
        
        # Load data
        training_data = load_data(vector_length, vector_method)
        
        print(f"Data loaded: {len(training_data)} samples")
        print(f"Model: {vector_length}→{bottleneck_size}→{vector_length}")
        
        # GPU Memory allocation
        training_data_array = np.array([item[0] for item in training_data], dtype=np.float32)
        
        # Set training data for PCA initialization (before weight initialization)
        gpu_ae.set_training_data_for_pca(training_data_array)
        print("Using PCA-based weight initialization")
        
        # Transfer training data to GPU
        gpu_ae.dataset_gpu = cuda.to_device(training_data_array)
        
        # Set training parameters
        gpu_ae.set_learning_rate(params['learning_rate'])
        gpu_ae.set_grad_clip_norm(params['grad_clip_norm'])
        
        print(f"Starting training with {epochs} epochs")
        print(f"Using batch size: {batch_size}")
        
        # Training
        best_loss = float('inf')
        best_loss_epoch = -1
        final_loss = None
        
        start_time = time.time()
        total_epochs = params['epochs']
        for epoch in range(total_epochs):
            # Random batch of indices
            batch_indices = np.random.choice(len(training_data), size=batch_size, replace=False)
            
            # Training step
            loss = gpu_ae.train_batch(batch_indices)
            
            # Test reconstruction quality periodically
            if epoch % 10 == 0:
              print(f"Epoch {epoch}: Loss = {loss:.6f}")

            if epoch % 10000 == 0 or epoch == 0 or epoch == total_epochs - 1:
                save_model(gpu_ae, training_data_array, training_data, best_loss, best_loss_epoch, final_loss, total_epochs, params, vector_method, start_time, epoch)
        
        gpu_ae.cleanup_gpu_memory()
        
        return best_loss < 0.01
    
    smoke_test_with_args(args.vector_method, args.bottleneck_size, args.vector_length, args.epochs, args.learning_rate, args.batch_size, args.max_home_distance, args.position_weight) 