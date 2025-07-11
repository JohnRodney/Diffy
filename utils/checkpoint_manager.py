#!/usr/bin/env python3
"""
Checkpoint management utility for saving, loading, and managing training checkpoints
"""

import os
import json
import datetime
import numpy as np

def save_checkpoint_metadata(checkpoint_path, epoch, loss, accuracy, session_id, training_params, start_time=None, elapsed_time=None, vector_drift_data=None):
    """Save training metadata to JSON file alongside model checkpoint"""
    json_filename = checkpoint_path.replace('.npz', '.json')
    
    metadata = {
        "session_info": {
            "session_id": session_id,
            "start_time": start_time.isoformat() if start_time else None,
            "elapsed_time_seconds": elapsed_time,
            "checkpoint_time": datetime.datetime.now().isoformat()
        },
        "model_architecture": {
            "vector_length": training_params["vector_length"],
            "hidden_layer_count": training_params["hidden_layer_count"],
            "bottleneck_size": training_params["bottleneck_size"],
            "leaky_relu_alpha": training_params["leaky_relu_alpha"],
            "total_colors": len(training_params["words"])
        },
        "training_hyperparameters": {
            "learning_rate": training_params["learning_rate"],
            "batch_size": training_params["batch_size"],
            "grad_clip_norm": training_params["grad_clip_norm"],
            "checkpoint_interval": training_params["checkpoint_interval"],
            "target_epochs": training_params["num_epochs"]
        },
        "training_progress": {
            "current_epoch": epoch,
            "loss": float(loss),
            "accuracy_percent": float(accuracy),
            "colors_learned": int(accuracy * len(training_params["words"]) / 100),
            "colors_total": len(training_params["words"])
        },
        "checkpoint_info": {
            "model_file": os.path.basename(checkpoint_path),
            "file_size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024) if os.path.exists(checkpoint_path) else 0
        }
    }
    
    # Add vector drift data if provided
    if vector_drift_data:
        metadata["vector_drift"] = vector_drift_data
    
    with open(json_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {json_filename}")
    return json_filename

def save_checkpoint(network, epoch, loss, accuracy, checkpoint_dir="memorycapacityruns", session_id=None, training_params=None, start_time=None, elapsed_time=None, vector_drift_data=None):
    """Save model checkpoint with training metadata"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename with session ID, epoch and metrics
    if session_id:
        checkpoint_filename = f"{session_id}_epoch_{epoch:06d}_loss_{loss:.6f}_acc_{accuracy:.1f}.npz"
    else:
        checkpoint_filename = f"checkpoint_epoch_{epoch:06d}_loss_{loss:.6f}_acc_{accuracy:.1f}.npz"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Save model parameters plus training metadata
    model_params = {}
    for i, layer in enumerate(network.all_layers):
        model_params[f"layer_{i}_weights"] = layer.weights
        model_params[f"layer_{i}_biases"] = layer.biases
    
    # Add training metadata
    model_params["epoch"] = epoch
    model_params["loss"] = loss
    model_params["accuracy"] = accuracy
    
    np.savez(checkpoint_path, **model_params)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save JSON metadata if training parameters provided
    if training_params:
        save_checkpoint_metadata(checkpoint_path, epoch, loss, accuracy, session_id, training_params, start_time, elapsed_time, vector_drift_data)
    
    return checkpoint_path

# Removed all deletion/cleanup functions per user request

def load_checkpoint(network, checkpoint_path):
    """Load model weights from checkpoint file"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        return False
    
    try:
        loaded_params = np.load(checkpoint_path)
        
        # Load model weights
        for i, layer in enumerate(network.all_layers):
            weight_key = f"layer_{i}_weights"
            bias_key = f"layer_{i}_biases"
            if weight_key in loaded_params and bias_key in loaded_params:
                layer.weights = loaded_params[weight_key]
                layer.biases = loaded_params[bias_key]
            else:
                print(f"Warning: Parameters for layer {i} not found in {checkpoint_path}.")
                return False
        
        # Load training metadata if available
        epoch = loaded_params.get("epoch", 0)
        loss = loaded_params.get("loss", 0.0)
        accuracy = loaded_params.get("accuracy", 0.0)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {epoch}, Loss: {loss:.6f}, Accuracy: {accuracy:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return False

def list_checkpoints(checkpoint_dir="memorycapacityruns", session_id=None):
    """List all available checkpoints for a session"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' not found.")
        return []
    
    # Get checkpoint files for this session
    if session_id:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{session_id}_epoch_") and f.endswith(".npz")]
    else:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".npz")]
    
    if not checkpoint_files:
        print(f"No checkpoints found for session: {session_id}")
        return []
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split("_epoch_")[1].split("_")[0]))
    
    print(f"\nAvailable checkpoints in {checkpoint_dir}:")
    print(f"{'Filename':<50} {'Epoch':<8} {'Loss':<12} {'Accuracy':<10}")
    print(f"{'='*50} {'='*8} {'='*12} {'='*10}")
    
    checkpoints_info = []
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            # Extract info from filename
            parts = filename.split("_")
            epoch = int(parts[2])
            loss = float(parts[4])
            accuracy = float(parts[5].replace(".npz", ""))
            
            print(f"{filename:<50} {epoch:<8} {loss:<12.6f} {accuracy:<10.1f}")
            
            checkpoints_info.append({
                'filename': filename,
                'filepath': filepath,
                'epoch': epoch,
                'loss': loss,
                'accuracy': accuracy
            })
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing {filename}: {e}")
    
    return checkpoints_info

def get_best_checkpoint(checkpoint_dir="memorycapacityruns", session_id=None, metric="accuracy"):
    """Get the best checkpoint based on specified metric"""
    checkpoints = list_checkpoints(checkpoint_dir, session_id)
    
    if not checkpoints:
        return None
    
    if metric == "accuracy":
        best_checkpoint = max(checkpoints, key=lambda x: x['accuracy'])
        print(f"\nBest checkpoint by accuracy: {best_checkpoint['filename']} ({best_checkpoint['accuracy']:.1f}%)")
    elif metric == "loss":
        best_checkpoint = min(checkpoints, key=lambda x: x['loss'])
        print(f"\nBest checkpoint by loss: {best_checkpoint['filename']} ({best_checkpoint['loss']:.6f})")
    else:
        print(f"Unknown metric: {metric}")
        return None
    
    return best_checkpoint 