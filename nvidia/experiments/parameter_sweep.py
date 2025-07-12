#!/usr/bin/env python3
"""
Simple smoke test for GPU autoencoder
"""

import numpy as np
import os
from gpu_networks.gpu_text_autoencoder import GPUTextAutoencoder
from color_loader import load_colors_from_json
from tokenizer.text_tokenizer import Tokenizer
from numba import cuda

def load_data(vector_length):
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
    
    tokenizer = Tokenizer(vocab_size=len(color_names))
    tokenizer.fill_dictionary(color_names)
    
    training_data = []
    for name in color_names:
        tokens = tokenizer.tokenize(name)
        encoded = np.zeros(vector_length, dtype=np.float32)
        for i, token in enumerate(tokens[:vector_length]):
            encoded[i] = float(token)
        training_data.append((encoded, name))
    
    return training_data

def smoke_test():
    """Simple smoke test with real data"""
    
    # Simple test parameters
    params = {
        'vector_length': 512,
        'hidden_layer_count': 2,
        'bottleneck_size': 128,
        'learning_rate': 0.001,
        'leaky_relu_alpha': 0.01,
        'batch_size': 256,
        'grad_clip_norm': 1
    }
    
    # Create model
    gpu_ae = GPUTextAutoencoder(
        vector_length=params['vector_length'],
        hidden_layer_count=params['hidden_layer_count'],
        bottleneck_size=params['bottleneck_size'],
        alpha=params['leaky_relu_alpha']
    )
            
    # Load real data
    training_data = load_data(params['vector_length'])
    training_data_array = np.array([item[0] for item in training_data], dtype=np.float32)
    
    # Keep tokenizer reference for saving dictionary
    colors_data = load_colors_from_json("colors_a_f.json")
    tokenizer = Tokenizer(vocab_size=len(colors_data))
    tokenizer.fill_dictionary(colors_data)
    
    # Set hyperparameters in the model
    gpu_ae.set_learning_rate(params['learning_rate'])
    gpu_ae.set_grad_clip_norm(params['grad_clip_norm'])
    
    # Load dataset into GPU memory
    gpu_ae.dataset_gpu = cuda.to_device(training_data_array)
    gpu_ae.dataset_size = len(training_data_array)
    
    print(f"Training on {len(training_data_array)} color vectors")
        
        # Training loop
    best_loss = float('inf')
    batch_size = min(params['batch_size'], len(training_data_array))
    
    for epoch in range(35000):
        indices = np.random.permutation(len(training_data_array))
        
        # Train on all batches using indices
        for i in range(0, len(training_data_array), batch_size):
            batch_indices = indices[i:i + batch_size]
            gpu_ae.train_batch(batch_indices.tolist())
        
        # Check loss every 50 epochs
        if epoch % 50 == 0:
            # Compute loss manually for now
            sample_indices = indices[:batch_size].tolist()
            sample_batch = cuda.device_array((len(sample_indices), gpu_ae.dataset_gpu.shape[1]), dtype=np.float32)
            for j, idx in enumerate(sample_indices):
                sample_batch[j] = gpu_ae.dataset_gpu[idx]
            output = gpu_ae.infer(sample_batch.copy_to_host())
            input_data = sample_batch.copy_to_host()
            loss = float(np.mean((input_data - output)**2))
            
            if loss < best_loss:
                best_loss = loss
            
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    # Save best model
    os.makedirs("/models", exist_ok=True)
    weights_cpu, biases_cpu = gpu_ae.export_weights()
    
    save_dict = {}
    for i, (w, b) in enumerate(zip(weights_cpu, biases_cpu)):
        save_dict[f'weight_{i}'] = w
        save_dict[f'bias_{i}'] = b
                    
    save_dict['training_vectors'] = training_data_array
    save_dict['training_names'] = [item[1] for item in training_data]
    save_dict['final_loss'] = best_loss
    save_dict['parameters'] = params
    save_dict['tokenizer_vocab'] = tokenizer.vocab  # Save the dictionary!
    
    np.savez_compressed("/app/models/smoke_test_model.npz", **save_dict)
    
    print(f"Final loss: {best_loss:.6f}")
    print("Model saved to /app/models/smoke_test_model.npz")
    
    # Clean up GPU memory when completely done
    gpu_ae.cleanup_gpu_memory()
    
    return best_loss < 0.01  # Success if loss is reasonable


if __name__ == "__main__":
    smoke_test() 