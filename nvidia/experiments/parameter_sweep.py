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
from datetime import datetime, timedelta
import sys

# Add paths for our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gpu_networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from gpu_text_autoencoder import GPUTextAutoencoder
except ImportError:
    print("Warning: gpu_text_autoencoder not available")
    GPUTextAutoencoder = None

try:
    from data_loader import load_color_data
except ImportError:
    print("Warning: data_loader not available")
    load_color_data = None

class ParameterSweep:
    """
    Automated parameter sweep for GPU training
    Handles experiment queueing, execution, and results collection
    """
    
    def __init__(self, base_output_dir=None):
        # Use mounted volume paths if running in Docker, local paths otherwise
        if base_output_dir is None:
            if os.path.exists("/app/logs"):  # Docker environment
                self.base_output_dir = "/app/logs/parameter_sweeps"
                self.models_dir = "/app/models" 
                self.results_dir = "/app/generated_images"  # Will contain our sweep results
            else:  # Local environment
                self.base_output_dir = "sweep_results"
                self.models_dir = "models"
                self.results_dir = "results"
        else:
            self.base_output_dir = base_output_dir
            self.models_dir = os.path.join(base_output_dir, "models")
            self.results_dir = os.path.join(base_output_dir, "results")
        
        self.results = []
        self.current_experiment = 0
        self.start_time = None
        
        # Create all necessary directories
        for dir_path in [self.base_output_dir, self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create session-based directory for this sweep
        self.session_dir = os.path.join(
            self.base_output_dir, 
            f"sweep_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"🚀 Parameter sweep initialized")
        print(f"📁 Session directory: {self.session_dir}")
        print(f"🏗️  Models directory: {self.models_dir}")
        print(f"📊 Results directory: {self.results_dir}")
    
    def define_parameter_grid(self):
        """
        Define the parameter space to explore
        Customize this based on your research questions
        """
        param_grid = {
            'vector_length': [128],  # Keep constant for now
            'hidden_layer_count': [1, 2, 3, 4],
            'bottleneck_size': [16, 32, 56, 64, 96],
            'learning_rate': [0.001, 0.003, 0.01, 0.03],
            'leaky_relu_alpha': [0.01, 0.05, 0.1, 0.2],
            'batch_size': [8, 16, 32],
            'grad_clip_norm': [0.5, 1.0, 2.0]
        }
        
        # Generate all combinations
        param_combinations = []
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)
        
        print(f"📊 Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def estimate_experiment_time(self, params, num_epochs=10000):
        """
        Estimate how long each experiment will take
        """
        # Rough estimates based on architecture size
        base_time_per_epoch = 0.01  # seconds
        
        # Factor in complexity
        complexity_factor = (
            params['vector_length'] * 
            params['hidden_layer_count'] * 
            params['batch_size']
        ) / 10000  # Normalize
        
        estimated_time = base_time_per_epoch * complexity_factor * num_epochs
        return estimated_time
    
    def run_single_experiment(self, params, experiment_id, max_epochs=10000):
        """
        Run a single training experiment
        """
        print(f"\n🔬 Starting experiment {experiment_id}")
        print(f"📋 Parameters: {params}")
        
        experiment_start = time.time()
        
        # Create experiment directory
        exp_dir = os.path.join(self.session_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save parameters
        param_file = os.path.join(exp_dir, "parameters.json")
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        try:
            # Create GPU autoencoder
            if GPUTextAutoencoder is None:
                raise ImportError("GPUTextAutoencoder not available")
            
            gpu_ae = GPUTextAutoencoder(
                vector_length=params['vector_length'],
                hidden_layer_count=params['hidden_layer_count'],
                bottleneck_size=params['bottleneck_size'],
                alpha=params['leaky_relu_alpha']
            )
            
            # Load training data (you'll need to implement this)
            # For now, create dummy data
            num_colors = 264
            training_data = self.create_dummy_training_data(
                num_colors, params['vector_length']
            )
            
            # Training loop
            training_history = []
            best_loss = float('inf')
            patience = 1000  # Early stopping patience
            no_improvement_count = 0
            
            for epoch in range(max_epochs):
                epoch_start = time.time()
                
                # Train on all batches
                total_loss = 0
                num_batches = 0
                
                for batch_start in range(0, len(training_data), params['batch_size']):
                    batch_end = min(batch_start + params['batch_size'], len(training_data))
                    batch_data = training_data[batch_start:batch_end]
                    
                    # Convert to numpy arrays
                    input_batch = np.array([item[0] for item in batch_data])
                    target_batch = np.array([item[1] for item in batch_data])
                    
                    # Train batch
                    batch_loss = gpu_ae.train_batch_gpu(
                        input_batch, target_batch,
                        params['learning_rate'], params['grad_clip_norm']
                    )
                    
                    total_loss += batch_loss
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                epoch_time = time.time() - epoch_start
                
                # Log progress
                training_history.append({
                    'epoch': epoch,
                    'loss': float(avg_loss),
                    'time': epoch_time,
                    'timestamp': time.time()
                })
                
                # Check for improvement
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improvement_count = 0
                    
                    # Save best model (both in experiment dir and models volume)
                    weights_cpu, biases_cpu = gpu_ae.get_weights_cpu()
                    
                    # Save to experiment directory for detailed tracking
                    exp_model_path = os.path.join(exp_dir, "best_model.npz")
                    np.savez(
                        exp_model_path,
                        weights=weights_cpu,
                        biases=biases_cpu,
                        loss=avg_loss,
                        epoch=epoch
                    )
                    
                    # Save to persistent models volume for easy access
                    global_model_path = os.path.join(
                        self.models_dir, 
                        f"exp_{experiment_id:04d}_best_model.npz"
                    )
                    np.savez(
                        global_model_path,
                        weights=weights_cpu,
                        biases=biases_cpu,
                        loss=avg_loss,
                        epoch=epoch,
                        parameters=params  # Include params for easy identification
                    )
                else:
                    no_improvement_count += 1
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}, Time = {epoch_time:.3f}s")
                
                # Early stopping
                if no_improvement_count >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
                
                # Save training history periodically
                if epoch % 500 == 0:
                    history_file = os.path.join(exp_dir, "training_history.json")
                    with open(history_file, 'w') as f:
                        json.dump(training_history, f, indent=2)
            
            # Final results
            experiment_time = time.time() - experiment_start
            
            result = {
                'experiment_id': experiment_id,
                'parameters': params,
                'best_loss': float(best_loss),
                'total_epochs': len(training_history),
                'experiment_time': experiment_time,
                'final_loss': float(training_history[-1]['loss']) if training_history else None,
                'converged': no_improvement_count < patience,
                'training_history': training_history
            }
            
            # Save detailed results
            result_file = os.path.join(exp_dir, "results.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✅ Experiment {experiment_id} completed")
            print(f"   Best loss: {best_loss:.6f}")
            print(f"   Total time: {experiment_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Experiment {experiment_id} failed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'parameters': params,
                'error': str(e),
                'status': 'failed'
            }
    
    def create_dummy_training_data(self, num_items, vector_length):
        """
        Create dummy training data for testing
        Replace with actual color data loading
        """
        training_data = []
        for i in range(num_items):
            # Create random vector (representing color)
            vector = np.random.randn(vector_length).astype(np.float32)
            # Autoencoder task: input = target
            training_data.append((vector, vector))
        return training_data
    
    def run_time_limited_sweep(self, max_runtime_hours=20):
        """
        Run parameter sweep with time limit
        Perfect for 20-hour server jobs
        """
        print(f"🕒 Starting {max_runtime_hours}-hour parameter sweep")
        
        self.start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600
        
        # Generate parameter combinations
        param_combinations = self.define_parameter_grid()
        
        # Estimate how many experiments we can fit
        sample_params = param_combinations[0]
        estimated_time_per_exp = self.estimate_experiment_time(sample_params)
        estimated_max_experiments = int(max_runtime_seconds / estimated_time_per_exp)
        
        print(f"📊 Estimated {estimated_max_experiments} experiments possible")
        
        # Run experiments
        experiment_id = 0
        for params in param_combinations:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Check if we have time for another experiment
            if elapsed_time + estimated_time_per_exp > max_runtime_seconds:
                print(f"⏰ Time limit reached. Stopping after {experiment_id} experiments.")
                break
            
            # Run experiment
            result = self.run_single_experiment(params, experiment_id)
            self.results.append(result)
            
            # Save aggregate results
            self.save_aggregate_results()
            
            experiment_id += 1
            
            # Print progress
            remaining_time = max_runtime_seconds - elapsed_time
            print(f"⏳ Time remaining: {remaining_time/3600:.1f} hours")
        
        print(f"🎉 Parameter sweep completed!")
        print(f"📈 Total experiments: {len(self.results)}")
        self.analyze_results()
        self.create_downloadable_summary()
    
    def save_aggregate_results(self):
        """
        Save all results to a master file
        """
        aggregate_file = os.path.join(self.session_dir, "aggregate_results.json")
        with open(aggregate_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def analyze_results(self):
        """
        Analyze and summarize results
        """
        if not self.results:
            return
        
        # Filter successful experiments
        successful_results = [r for r in self.results if 'best_loss' in r]
        
        if not successful_results:
            print("❌ No successful experiments")
            return
        
        # Find best result
        best_result = min(successful_results, key=lambda x: x['best_loss'])
        
        print(f"\n🏆 BEST RESULT:")
        print(f"   Loss: {best_result['best_loss']:.6f}")
        print(f"   Parameters: {best_result['parameters']}")
        
        # Summary statistics
        losses = [r['best_loss'] for r in successful_results]
        print(f"\n📊 SUMMARY:")
        print(f"   Successful experiments: {len(successful_results)}")
        print(f"   Best loss: {min(losses):.6f}")
        print(f"   Worst loss: {max(losses):.6f}")
        print(f"   Average loss: {np.mean(losses):.6f}")
        
        # Save analysis
        analysis = {
            'best_result': best_result,
            'summary_stats': {
                'successful_experiments': len(successful_results),
                'best_loss': float(min(losses)),
                'worst_loss': float(max(losses)),
                'average_loss': float(np.mean(losses))
            }
        }
        
        analysis_file = os.path.join(self.session_dir, "analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def create_downloadable_summary(self):
        """
        Create summary files in the results volume for easy download
        """
        print(f"\n📦 Creating downloadable summary...")
        
        # Create summary directory in results volume
        summary_dir = os.path.join(self.results_dir, f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Copy key files to summary directory
        summary_files = [
            ("aggregate_results.json", "all_experiment_results.json"),
            ("analysis.json", "best_results_analysis.json")
        ]
        
        for src_name, dst_name in summary_files:
            src_path = os.path.join(self.session_dir, src_name)
            dst_path = os.path.join(summary_dir, dst_name)
            if os.path.exists(src_path):
                with open(src_path, 'r') as src_file:
                    data = json.load(src_file)
                with open(dst_path, 'w') as dst_file:
                    json.dump(data, dst_file, indent=2)
        
        # Create a human-readable summary
        readable_summary = os.path.join(summary_dir, "README.md")
        with open(readable_summary, 'w') as f:
            f.write(f"# Parameter Sweep Results\n\n")
            f.write(f"**Session:** {os.path.basename(self.session_dir)}\n")
            f.write(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Experiments:** {len(self.results)}\n\n")
            
            # Add best result if available
            successful_results = [r for r in self.results if 'best_loss' in r]
            if successful_results:
                best_result = min(successful_results, key=lambda x: x['best_loss'])
                f.write(f"## 🏆 Best Result\n\n")
                f.write(f"**Loss:** {best_result['best_loss']:.6f}\n")
                f.write(f"**Parameters:**\n")
                for key, value in best_result['parameters'].items():
                    f.write(f"- {key}: {value}\n")
                f.write(f"\n")
            
            f.write(f"## 📁 Files\n\n")
            f.write(f"- `all_experiment_results.json` - Complete results from all experiments\n")
            f.write(f"- `best_results_analysis.json` - Analysis of best performing models\n")
            f.write(f"- Models are saved in `/app/models/` with format `exp_XXXX_best_model.npz`\n")
        
        print(f"✅ Summary created in: {summary_dir}")
        print(f"📁 This directory will be downloaded automatically by deploy script")

def main():
    """
    Main function for running parameter sweep
    """
    # Create parameter sweep
    sweep = ParameterSweep()
    
    # Run 20-hour sweep
    sweep.run_time_limited_sweep(max_runtime_hours=20)

if __name__ == "__main__":
    main() 