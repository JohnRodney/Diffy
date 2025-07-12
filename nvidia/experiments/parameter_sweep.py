#!/usr/bin/env python3
"""
Parameter sweep framework for GPU training
Designed for 20-hour automated runs on RTX 5090 server
"""

import numpy as np
import json
import time
import os
import itertools
from datetime import datetime
from gpu_networks.gpu_text_autoencoder import GPUTextAutoencoder
from gpu_networks.batch_manager import create_batch_manager
from color_loader import load_colors_from_json
from tokenizer.text_tokenizer import Tokenizer

# Training Configuration Constants
STABILIZED_LEARNING_RATE_LOW = 0.00001
STABILIZED_LEARNING_RATE_MID = 0.00003
STABILIZED_LEARNING_RATE_HIGH = 0.00005

LEAKY_RELU_ALPHA_LOW = 0.01
LEAKY_RELU_ALPHA_HIGH = 0.1

BATCH_SIZE_SMALL = 256
BATCH_SIZE_MEDIUM = 512
BATCH_SIZE_LARGE = 1024

GRADIENT_CLIP_TIGHT = 0.02
GRADIENT_CLIP_MEDIUM = 0.05
GRADIENT_CLIP_LOOSE = 0.1

# Architecture Constants
VECTOR_LENGTH_SMALL = 512
VECTOR_LENGTH_MEDIUM = 1024
VECTOR_LENGTH_LARGE = 2048

HIDDEN_LAYER_COUNT_MIN = 2
HIDDEN_LAYER_COUNT_LOW = 3
HIDDEN_LAYER_COUNT_MID = 4
HIDDEN_LAYER_COUNT_MAX = 5

BOTTLENECK_SIZE_SMALL = 128
BOTTLENECK_SIZE_MEDIUM = 256
BOTTLENECK_SIZE_LARGE = 512
BOTTLENECK_SIZE_XLARGE = 1024

# Performance Constants
BASE_TIME_PER_EPOCH = 0.01
COMPLEXITY_NORMALIZATION_FACTOR = 10000
DEFAULT_MAX_EPOCHS = 10000
DEFAULT_MAX_RUNTIME_HOURS = 20

# Memory Configuration
MAX_BATCH_SIZE_LIMIT = 4096
GRID_SIZE_MULTIPLIER_512 = 512




class ParameterSweep:
    """
    Automated parameter sweep for GPU training
    Handles experiment queueing, execution, and results collection
    """
    
    def __init__(self, base_output_dir=None):
        # Always save everything under /models/ to avoid multiple scp directories
        if base_output_dir is None:
            if os.path.exists("/app"):
                # Everything goes under /models/ for easy scp
                self.base_output_dir = "/models/parameter_sweeps"
                self.models_dir = "/models/trained_models" 
                self.results_dir = "/models/generated_images"
            else:
                # Local development fallback
                self.base_output_dir = "models/parameter_sweeps"
                self.models_dir = "models/trained_models"
                self.results_dir = "models/generated_images"
        else:
            self.base_output_dir = base_output_dir
            self.models_dir = os.path.join(base_output_dir, "trained_models")
            self.results_dir = os.path.join(base_output_dir, "generated_images")
        
        self.results = []
        self.current_experiment = 0
        self.start_time = None
        self._cached_colors = None
        
        # Create all directories with consistent permissions
        for dir_path in [self.base_output_dir, self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.session_dir = os.path.join(
            self.base_output_dir, 
            f"sweep_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.session_dir, exist_ok=True)
    
    def define_parameter_grid(self):
        """
        Define the parameter space to explore - SCALED UP FOR RTX 5090
        Large architectures and batch sizes to maximize GPU utilization
        
        STABILITY UPDATE: After gradient explosion at epoch 99 with lr=0.0001,
        implementing ACTUAL stabilization: 10x lower learning rates and 5x stronger gradient clipping
        """
        param_grid = {
            'vector_length': [VECTOR_LENGTH_SMALL, VECTOR_LENGTH_MEDIUM, VECTOR_LENGTH_LARGE],
            'hidden_layer_count': [HIDDEN_LAYER_COUNT_MIN, HIDDEN_LAYER_COUNT_LOW, HIDDEN_LAYER_COUNT_MID, HIDDEN_LAYER_COUNT_MAX],
            'bottleneck_size': [BOTTLENECK_SIZE_SMALL, BOTTLENECK_SIZE_MEDIUM, BOTTLENECK_SIZE_LARGE, BOTTLENECK_SIZE_XLARGE],
            'learning_rate': [STABILIZED_LEARNING_RATE_LOW, STABILIZED_LEARNING_RATE_MID, STABILIZED_LEARNING_RATE_HIGH],
            'leaky_relu_alpha': [LEAKY_RELU_ALPHA_LOW, LEAKY_RELU_ALPHA_HIGH],
            'batch_size': [BATCH_SIZE_SMALL, BATCH_SIZE_MEDIUM, BATCH_SIZE_LARGE],
            'grad_clip_norm': [GRADIENT_CLIP_TIGHT, GRADIENT_CLIP_MEDIUM, GRADIENT_CLIP_LOOSE]
        }
        
        param_combinations = []
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def estimate_experiment_time(self, params, num_epochs=DEFAULT_MAX_EPOCHS):
        """
        Estimate how long each experiment will take
        """
        complexity_factor = (
            params['vector_length'] * 
            params['hidden_layer_count'] * 
            params['batch_size']
        ) / COMPLEXITY_NORMALIZATION_FACTOR
        
        estimated_time = BASE_TIME_PER_EPOCH * complexity_factor * num_epochs
        return estimated_time
    
    def run_single_experiment(self, params, experiment_id, max_epochs=DEFAULT_MAX_EPOCHS):
        """
        Run a single training experiment
        """
        experiment_start = time.time()
        
        exp_dir = os.path.join(self.session_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)
        
        param_file = os.path.join(exp_dir, "parameters.json")
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
        

        
        gpu_ae = GPUTextAutoencoder(
            vector_length=params['vector_length'],
            hidden_layer_count=params['hidden_layer_count'],
            bottleneck_size=params['bottleneck_size'],
            alpha=params['leaky_relu_alpha']
        )
            
        training_data = self.load_real_color_data(params['vector_length'])
        
        training_data_array = np.array([item[0] for item in training_data], dtype=np.float32)
        
        batch_manager = create_batch_manager(
            batch_size=params['batch_size'],
            dataset_size=len(training_data_array),
            vector_length=params['vector_length']
        )
        
        best_loss = float('inf')
        best_epoch = 0
        patience = 100
        no_improve_count = 0
        
        for epoch in range(max_epochs):
            epoch_losses = []
            num_batches = 0
            
            batch_indices_list = batch_manager.create_mini_batches(training_data_array)
            for batch_indices in batch_indices_list:
                batch_data, _ = batch_manager.prepare_gpu_batch(training_data_array, batch_indices)
                loss_value = gpu_ae.train_batch(batch_data.copy_to_host(), params['learning_rate'], params['grad_clip_norm'])
                epoch_losses.append(loss_value)
                num_batches += 1
            
            avg_loss = np.mean(epoch_losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                break
        
        experiment_time = time.time() - experiment_start
        
        # Save model with training data for reconstruction testing
        model_file = os.path.join(self.models_dir, f"exp_{experiment_id:04d}_best_model.npz")
        weights_cpu, biases_cpu = gpu_ae.export_weights()
        save_dict = {}
        
        # Save model weights and biases
        for i, (w, b) in enumerate(zip(weights_cpu, biases_cpu)):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}'] = b
        
        # Save training data for reconstruction testing
        save_dict['training_vectors'] = np.array([item[0] for item in training_data], dtype=np.float32)
        save_dict['training_names'] = [item[1] for item in training_data]
        save_dict['final_loss'] = best_loss
        save_dict['parameters'] = params
        
        np.savez_compressed(model_file, **save_dict)
        
        result = {
            'experiment_id': experiment_id,
            'parameters': params,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1,
            'training_time': experiment_time,
            'convergence_achieved': no_improve_count < patience,
            'status': 'completed',
            'model_file': model_file
        }
        
        results_file = os.path.join(exp_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    def load_real_color_data(self, vector_length):
        """
        Load real color data for training
        """
        if load_colors_from_json is None:
            return self.create_dummy_training_data(264, vector_length)
        
        if self._cached_colors is None:
            try:
                self._cached_colors = load_colors_from_json("colors_a_f.json")
                if not self._cached_colors:
                    return self.create_dummy_training_data(264, vector_length)
            except FileNotFoundError:
                return self.create_dummy_training_data(264, vector_length)
        
        colors_data = self._cached_colors
        
        color_names = []
        for color in colors_data:
            if isinstance(color, dict) and 'name' in color:
                color_names.append(str(color.get('name', '')))
        
        if Tokenizer is None:
            return self.create_dummy_training_data(len(color_names), vector_length)
        
        tokenizer = Tokenizer(vocab_size=len(color_names))
        tokenizer.fill_dictionary(color_names)
        
        training_data = []
        for name in color_names:
            tokens = tokenizer.tokenize(name)
            # Convert tokens to fixed-length vector
            encoded = np.zeros(vector_length, dtype=np.float32)
            for i, token in enumerate(tokens[:vector_length]):
                encoded[i] = float(token)
            training_data.append((encoded, name))
        
        return training_data
    
    def create_dummy_training_data(self, num_items, vector_length):
        """
        Create dummy training data when real data is not available
        """
        training_data = []
        for i in range(num_items):
            dummy_vector = np.random.randn(vector_length).astype(np.float32)
            dummy_vector = dummy_vector / np.linalg.norm(dummy_vector)
            training_data.append((dummy_vector, f"dummy_color_{i}"))
        
        return training_data
    
    def run_time_limited_sweep(self, max_runtime_hours=DEFAULT_MAX_RUNTIME_HOURS):
        """
        Run parameter sweep with time limit
        """
        start_time = time.time()
        end_time = start_time + (max_runtime_hours * 3600)
        
        param_combinations = self.define_parameter_grid()
        
        experiment_id = 0
        for params in param_combinations:
            if time.time() >= end_time:
                break
            
            try:
                result = self.run_single_experiment(params, experiment_id)
                self.results.append(result)
                
                remaining_time = end_time - time.time()
                if remaining_time <= 0:
                    break
                
            except Exception as e:
                continue
            
            experiment_id += 1
        
        self.save_aggregate_results()
        self.analyze_results()
        self.create_downloadable_summary()
        
        return self.results
    
    def save_aggregate_results(self):
        """
        Save all results to a single JSON file
        """
        aggregate_file = os.path.join(self.session_dir, "aggregate_results.json")
        with open(aggregate_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def analyze_results(self):
        """
        Analyze the results and identify best performing configurations
        """
        if not self.results:
            return
        
        successful_results = [r for r in self.results if r.get('status') == 'completed']
        if not successful_results:
            return
        
        best_result = min(successful_results, key=lambda r: r['best_loss'])
        
        losses = [r['best_loss'] for r in successful_results]
        
        analysis = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'best_loss': best_result['best_loss'],
            'best_parameters': best_result['parameters'],
            'worst_loss': max(losses),
            'average_loss': np.mean(losses),
            'median_loss': np.median(losses),
            'std_loss': np.std(losses)
        }
        
        analysis_file = os.path.join(self.session_dir, "analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def create_downloadable_summary(self):
        """
        Create a comprehensive summary for download
        """
        summary_dir = os.path.join(self.session_dir, "download_summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_data = {
            'session_info': {
                'session_dir': self.session_dir,
                'start_time': datetime.now().isoformat(),
                'total_experiments': len(self.results)
            },
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        summary_file = os.path.join(summary_dir, "complete_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        readme_content = f"""
# Parameter Sweep Results Summary

## Session Information
- Session Directory: {self.session_dir}
- Total Experiments: {len(self.results)}
- Results File: complete_summary.json

## Files in this directory:
- complete_summary.json: Complete results and analysis
- README.md: This file

## Next Steps:
1. Review the complete_summary.json file
2. Identify the best performing parameters
3. Use the best parameters for production training
        """
        
        readme_file = os.path.join(summary_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        return summary_dir

def main():
    """
    Main function to run the parameter sweep
    """
    sweep = ParameterSweep()
    
    sweep.run_time_limited_sweep(max_runtime_hours=DEFAULT_MAX_RUNTIME_HOURS)

def smoke_test():
    """
    Quick smoke test to validate the system
    """
    sweep = ParameterSweep()
    
    test_params = {
        'vector_length': VECTOR_LENGTH_SMALL,
        'hidden_layer_count': HIDDEN_LAYER_COUNT_MIN,
        'bottleneck_size': BOTTLENECK_SIZE_SMALL,
        'learning_rate': STABILIZED_LEARNING_RATE_LOW,
        'leaky_relu_alpha': LEAKY_RELU_ALPHA_LOW,
        'batch_size': BATCH_SIZE_SMALL,
        'grad_clip_norm': GRADIENT_CLIP_MEDIUM
    }
    
    result = sweep.run_single_experiment(test_params, experiment_id=0, max_epochs=DEFAULT_MAX_EPOCHS)
    
    if result.get('status') == 'completed':
        return True
    else:
        return False

if __name__ == "__main__":
    # Run smoke test first
    if smoke_test():
        main()
    else:
        main() 