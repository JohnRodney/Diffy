#!/usr/bin/env python3
"""
Model persistence utility for saving and loading neural network models
"""

import os
import numpy as np

def save_model(network, filename="autoencoder_model.npz"):
    """Save model weights and biases to file"""
    model_params = {}
    for i, layer in enumerate(network.all_layers):
        model_params[f"layer_{i}_weights"] = layer.weights
        model_params[f"layer_{i}_biases"] = layer.biases
    
    np.savez(filename, **model_params)
    print(f"Model saved to {filename}")
    return filename

def load_model(network, filename="autoencoder_model.npz"):
    """Load model weights and biases from file"""
    if not os.path.exists(filename):
        print(f"Error: Model file '{filename}' not found.")
        return False
    
    try:
        loaded_params = np.load(filename)
        
        for i, layer in enumerate(network.all_layers):
            weight_key = f"layer_{i}_weights"
            bias_key = f"layer_{i}_biases"
            if weight_key in loaded_params and bias_key in loaded_params:
                layer.weights = loaded_params[weight_key]
                layer.biases = loaded_params[bias_key]
            else:
                print(f"Warning: Parameters for layer {i} not found in {filename}. Skipping.")
                return False
        
        print(f"Model loaded from {filename}")
        return True
        
    except Exception as e:
        print(f"Error loading model {filename}: {e}")
        return False

def get_model_info(filename):
    """Get information about a saved model file"""
    if not os.path.exists(filename):
        print(f"Error: Model file '{filename}' not found.")
        return None
    
    try:
        loaded_params = np.load(filename)
        
        # Count layers
        layer_count = 0
        while f"layer_{layer_count}_weights" in loaded_params:
            layer_count += 1
        
        # Get file size
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        
        # Get model architecture info
        first_layer_weights = loaded_params["layer_0_weights"]
        input_size = first_layer_weights.shape[0]
        
        # Find bottleneck size (smallest layer)
        layer_sizes = []
        for i in range(layer_count):
            weights = loaded_params[f"layer_{i}_weights"]
            layer_sizes.append(weights.shape[1])
        
        bottleneck_size = min(layer_sizes)
        
        model_info = {
            'filename': filename,
            'file_size_mb': file_size,
            'total_layers': layer_count,
            'input_size': input_size,
            'bottleneck_size': bottleneck_size,
            'layer_sizes': layer_sizes,
            'has_training_metadata': 'epoch' in loaded_params
        }
        
        if model_info['has_training_metadata']:
            model_info['epoch'] = int(loaded_params.get('epoch', 0))
            model_info['loss'] = float(loaded_params.get('loss', 0.0))
            model_info['accuracy'] = float(loaded_params.get('accuracy', 0.0))
        
        return model_info
        
    except Exception as e:
        print(f"Error reading model info from {filename}: {e}")
        return None

def list_model_files(directory="."):
    """List all model files in a directory"""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return []
    
    model_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    
    if not model_files:
        print(f"No model files found in {directory}")
        return []
    
    print(f"\nModel files in {directory}:")
    print(f"{'Filename':<40} {'Size (MB)':<10} {'Layers':<8} {'Bottleneck':<12} {'Epoch':<8} {'Accuracy':<10}")
    print(f"{'='*40} {'='*10} {'='*8} {'='*12} {'='*8} {'='*10}")
    
    models_info = []
    for filename in model_files:
        filepath = os.path.join(directory, filename)
        info = get_model_info(filepath)
        if info:
            epoch_str = str(info['epoch']) if info['has_training_metadata'] else "N/A"
            accuracy_str = f"{info['accuracy']:.1f}%" if info['has_training_metadata'] else "N/A"
            
            print(f"{filename:<40} {info['file_size_mb']:<10.1f} {info['total_layers']:<8} {info['bottleneck_size']:<12} {epoch_str:<8} {accuracy_str:<10}")
            models_info.append(info)
    
    return models_info

def compare_models(model_files):
    """Compare multiple model files"""
    if not model_files:
        print("No model files provided for comparison.")
        return []
    
    models_info = []
    for filename in model_files:
        info = get_model_info(filename)
        if info:
            models_info.append(info)
    
    if not models_info:
        print("No valid model files found.")
        return []
    
    print(f"\n=== Model Comparison ===")
    print(f"{'Model':<30} {'Accuracy':<10} {'Loss':<12} {'Epoch':<8} {'Size (MB)':<10}")
    print(f"{'='*30} {'='*10} {'='*12} {'='*8} {'='*10}")
    
    for info in models_info:
        filename = os.path.basename(info['filename'])
        if info['has_training_metadata']:
            print(f"{filename:<30} {info['accuracy']:<10.1f} {info['loss']:<12.6f} {info['epoch']:<8} {info['file_size_mb']:<10.1f}")
        else:
            print(f"{filename:<30} {'N/A':<10} {'N/A':<12} {'N/A':<8} {info['file_size_mb']:<10.1f}")
    
    # Find best model
    trained_models = [info for info in models_info if info['has_training_metadata']]
    if trained_models:
        best_accuracy = max(trained_models, key=lambda x: x['accuracy'])
        best_loss = min(trained_models, key=lambda x: x['loss'])
        
        print(f"\nBest by accuracy: {os.path.basename(best_accuracy['filename'])} ({best_accuracy['accuracy']:.1f}%)")
        print(f"Best by loss: {os.path.basename(best_loss['filename'])} ({best_loss['loss']:.6f})")
    
    return models_info

def backup_model(source_file, backup_dir="backups"):
    """Create a backup of a model file"""
    if not os.path.exists(source_file):
        print(f"Source file '{source_file}' not found.")
        return False
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create backup filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.basename(source_file)
    name, ext = os.path.splitext(source_name)
    backup_filename = f"{name}_backup_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        import shutil
        shutil.copy2(source_file, backup_path)
        print(f"Model backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False 