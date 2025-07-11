#!/usr/bin/env python3
"""
Analyze downloaded model files from parameter sweep
"""
import os
import sys

def analyze_models():
    """Analyze model files in downloaded_models directory"""
    model_dir = "downloaded_models"
    
    if not os.path.exists(model_dir):
        print("❌ downloaded_models directory not found")
        return
    
    print("🔍 Model Analysis")
    print("=" * 50)
    
    # Get all NPZ files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.npz')]
    model_files.sort()
    
    print(f"📊 Found {len(model_files)} model files")
    print()
    
    # Analyze file sizes and patterns
    for filename in model_files:
        filepath = os.path.join(model_dir, filename)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        
        # Extract experiment number
        exp_num = filename.split('_')[1]
        
        print(f"📁 {filename}")
        print(f"   Experiment: {exp_num}")
        print(f"   Size: {file_size:.1f} MB")
        
        # Try to load with numpy if available
        try:
            import numpy as np
            with np.load(filepath) as data:
                print(f"   Keys: {list(data.keys())}")
                
                # Analyze weight shapes to infer architecture
                weights = [key for key in data.keys() if 'weight' in key.lower()]
                if weights:
                    print(f"   Weight layers: {len(weights)}")
                    # Check first weight to estimate input size
                    first_weight = data[weights[0]]
                    print(f"   Input size: {first_weight.shape[0]}")
                    print(f"   First layer: {first_weight.shape}")
                    
        except ImportError:
            print("   (numpy not available - can't analyze structure)")
        except Exception as e:
            print(f"   Error loading: {e}")
            
        print()

if __name__ == "__main__":
    analyze_models() 