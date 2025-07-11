#!/usr/bin/env python3
"""
Extract parameters from model NPZ files
The parameters are stored inside each model file!
"""

import numpy as np
import json
import os

def extract_params_from_models():
    """Extract parameters from all downloaded model files"""
    models_dir = "downloaded_models"
    
    if not os.path.exists(models_dir):
        print("❌ downloaded_models directory not found")
        return
    
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.npz')])
    
    print("🔍 Extracting Parameters from Model Files")
    print("=" * 60)
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        try:
            # Load model
            model_data = np.load(model_path, allow_pickle=True)
            
            print(f"\n📁 {model_file}")
            print(f"   Keys: {list(model_data.keys())}")
            print(f"   Loss: {model_data.get('loss', 'N/A')}")
            print(f"   Epoch: {model_data.get('epoch', 'N/A')}")
            
            # Extract parameters if they exist
            if 'parameters' in model_data:
                params = model_data['parameters'].item()  # .item() to convert numpy array to dict
                print(f"   📋 Parameters:")
                for key, value in params.items():
                    print(f"      {key}: {value}")
                
                # Save parameters to separate JSON file for easy access
                param_file = os.path.join(models_dir, f"{model_file.replace('.npz', '')}_params.json")
                with open(param_file, 'w') as f:
                    json.dump(params, f, indent=2)
                print(f"   ✅ Parameters saved to {param_file}")
                
            else:
                print(f"   ❌ No parameters found in model file")
                
        except Exception as e:
            print(f"   ❌ Error loading {model_file}: {e}")
    
    print(f"\n✅ Parameter extraction complete!")

def analyze_parameter_patterns():
    """Analyze the parameter patterns across models"""
    models_dir = "downloaded_models"
    param_files = [f for f in os.listdir(models_dir) if f.endswith('_params.json')]
    
    if not param_files:
        print("❌ No parameter files found - run extract_params_from_models() first")
        return
    
    print("\n🔍 Parameter Pattern Analysis")
    print("=" * 60)
    
    all_params = []
    
    for param_file in sorted(param_files):
        param_path = os.path.join(models_dir, param_file)
        
        with open(param_path, 'r') as f:
            params = json.load(f)
            all_params.append((param_file, params))
    
    # Group by key parameters
    by_vector_length = {}
    by_architecture = {}
    
    for filename, params in all_params:
        exp_id = filename.split('_')[1]
        
        # Group by vector length
        vl = params['vector_length']
        if vl not in by_vector_length:
            by_vector_length[vl] = []
        by_vector_length[vl].append((exp_id, params))
        
        # Group by architecture
        arch = f"{params['hidden_layer_count']}L_{params['bottleneck_size']}B"
        if arch not in by_architecture:
            by_architecture[arch] = []
        by_architecture[arch].append((exp_id, params))
    
    print("\n📊 Experiments by Vector Length:")
    for vl, experiments in by_vector_length.items():
        print(f"   {vl} dims: {len(experiments)} experiments")
        for exp_id, params in experiments:
            print(f"      exp_{exp_id}: {params['hidden_layer_count']}L, {params['bottleneck_size']}B, "
                  f"lr={params['learning_rate']}, bs={params['batch_size']}")
    
    print("\n🏗️  Experiments by Architecture:")
    for arch, experiments in by_architecture.items():
        print(f"   {arch}: {len(experiments)} experiments")
        for exp_id, params in experiments:
            print(f"      exp_{exp_id}: {params['vector_length']}d, lr={params['learning_rate']}, "
                  f"bs={params['batch_size']}")

def main():
    """Main function"""
    extract_params_from_models()
    analyze_parameter_patterns()

if __name__ == "__main__":
    main() 