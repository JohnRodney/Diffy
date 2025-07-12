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
            for i, token in enumerate(tokens[:vector_length]):
                encoded[i] = float(token)
            training_data.append((encoded, name))
    else:
        # Create random vectors for each color name (like CPU version)
        for name in color_names:
            # Generate unique random vector for this color
            encoded = np.random.rand(vector_length).astype(np.float32)
            training_data.append((encoded, name))
    
    return training_data

def smoke_test(vector_method="random"):
    """Simple smoke test with real data"""
    
    # Record start time
    start_time = time.time()
    
    # Simple test parameters
    params = {
        'vector_length': 512,
        'hidden_layer_count': 2,
        'bottleneck_size': 128,
        'learning_rate': 0.001,
        'leaky_relu_alpha': 0.01,
        'batch_size': 256,
        'grad_clip_norm': 1,
        "epochs": 10000,
    }
    
    # Create model
    gpu_ae = GPUTextAutoencoder(
        vector_length=params['vector_length'],
        hidden_layer_count=params['hidden_layer_count'],
        bottleneck_size=params['bottleneck_size'],
        alpha=params['leaky_relu_alpha']
    )
            
    # Load real data
    training_data = load_data(params['vector_length'], vector_method)
    training_data_array = np.array([item[0] for item in training_data], dtype=np.float32)
    
    # Calculate and report similarity for research purposes
    mean_similarity = 0.0
    if len(training_data_array) > 1:
        similarity_sum = 0
        count = 0
        for i in range(len(training_data_array)):
            for j in range(i + 1, len(training_data_array)):
                dot_product = np.dot(training_data_array[i], training_data_array[j])
                norm_i = np.linalg.norm(training_data_array[i])
                norm_j = np.linalg.norm(training_data_array[j])
                if norm_i > 0 and norm_j > 0:
                    similarity_sum += dot_product / (norm_i * norm_j)
                    count += 1
        mean_similarity = similarity_sum / count if count > 0 else 0
        print(f"Mean cosine similarity ({vector_method}): {mean_similarity:.6f}")
    
    # Set hyperparameters in the model
    gpu_ae.set_learning_rate(params['learning_rate'])
    gpu_ae.set_grad_clip_norm(params['grad_clip_norm'])
    
    # Load dataset into GPU memory
    gpu_ae.dataset_gpu = cuda.to_device(training_data_array)
    gpu_ae.dataset_size = len(training_data_array)
    
    print(f"Training on {len(training_data_array)} color vectors")
    
    # Training setup
    total_epochs = params['epochs']
    best_loss = float('inf')
    best_loss_epoch = 0
    final_loss = float('inf')
    batch_size = min(params['batch_size'], len(training_data_array))
    
    # Training loop
    for epoch in range(total_epochs):
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
            # Cosine similarity loss: 1 - cosine_similarity
            cos_sim = np.sum(input_data * output, axis=1) / (np.linalg.norm(input_data, axis=1) * np.linalg.norm(output, axis=1))
            loss = float(np.mean(1 - cos_sim))
            final_loss = loss  # Track the most recent loss
            
            if loss < best_loss:
                best_loss = loss
                best_loss_epoch = epoch
                print(f"Epoch {epoch}: Loss = {loss:.6f} (NEW BEST)")
            else:
                print(f"Epoch {epoch}: Loss = {loss:.6f} (best: {best_loss:.6f} at epoch {best_loss_epoch})")
    
    # Calculate training duration
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Save best model
    os.makedirs("/models", exist_ok=True)
    weights_cpu, biases_cpu = gpu_ae.export_weights()
    
    # Get REAL model outputs for analysis (while CUDA actually works!)
    print("Saving real model outputs for analysis...")
    real_model_outputs = gpu_ae.infer(training_data_array)
    
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
    save_dict['mean_cosine_similarity'] = mean_similarity
    save_dict['training_start_time'] = start_time
    save_dict['training_end_time'] = end_time
    save_dict['training_duration_seconds'] = training_duration
    save_dict['epochs_per_second'] = total_epochs / training_duration
    save_dict['dataset_size'] = len(training_data_array)
    
    np.savez_compressed("/app/models/smoke_test_model.npz", **save_dict)
    
    print(f"Best loss: {best_loss:.6f} at epoch {best_loss_epoch}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Training duration: {training_duration:.2f} seconds")
    print(f"Epochs per second: {total_epochs / training_duration:.2f}")
    print("Model saved to /app/models/smoke_test_model.npz")
    
    # Clean up GPU memory when completely done
    gpu_ae.cleanup_gpu_memory()
    
    return best_loss < 0.01  # Success if loss is reasonable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU autoencoder smoke test")
    parser.add_argument("--vector-method", choices=["random", "tokenized"], default="random",
                        help="Method for generating vectors: random (like CPU version) or tokenized (research bug)")
    args = parser.parse_args()
    
    print(f"Using vector method: {args.vector_method}")
    smoke_test(args.vector_method) 