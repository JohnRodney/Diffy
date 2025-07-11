#!/usr/bin/env python3
"""
Training analysis utility for loading and analyzing training progression data
"""

import json
import os

def load_training_progression(checkpoint_dir="memorycapacityruns", session_id=None):
    """Load training progression from JSON metadata files"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' not found.")
        return []
    
    # Get all JSON metadata files for this session
    if session_id:
        json_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{session_id}_epoch_") and f.endswith(".json")]
    else:
        json_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".json")]
    
    if not json_files:
        print(f"No training metadata found for session: {session_id}")
        return []
    
    # Sort by epoch number
    json_files.sort(key=lambda x: int(x.split("_epoch_")[1].split("_")[0]))
    
    progression = []
    for filename in json_files:
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)
                progression.append({
                    'epoch': metadata['training_progress']['current_epoch'],
                    'loss': metadata['training_progress']['loss'],
                    'accuracy': metadata['training_progress']['accuracy_percent'],
                    'colors_learned': metadata['training_progress']['colors_learned'],
                    'elapsed_time': metadata['session_info']['elapsed_time_seconds'],
                    'vector_drift': metadata.get('vector_drift', None)
                })
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error reading {filename}: {e}")
    
    return progression

def analyze_training_progression(checkpoint_dir="memorycapacityruns", session_id=None):
    """Analyze training progression from saved JSON files"""
    progression = load_training_progression(checkpoint_dir, session_id)
    
    if not progression:
        print("No training progression data found.")
        return
    
    print(f"\n=== Training Progression Analysis ===")
    print(f"Session: {session_id}")
    print(f"Total checkpoints: {len(progression)}")
    print(f"Epochs covered: {progression[0]['epoch']} to {progression[-1]['epoch']}")
    print(f"Training time: {progression[-1]['elapsed_time']:.1f} seconds")
    
    print(f"\n{'Epoch':<8} {'Loss':<10} {'Accuracy':<10} {'Colors':<8} {'Time':<10}")
    print(f"{'='*8} {'='*10} {'='*10} {'='*8} {'='*10}")
    
    for data in progression:
        print(f"{data['epoch']:<8} {data['loss']:<10.6f} {data['accuracy']:<10.1f} {data['colors_learned']:<8} {data['elapsed_time']:<10.1f}")
    
    # Show learning curve insights
    if len(progression) > 1:
        print(f"\n=== Learning Curve Analysis ===")
        initial_loss = progression[0]['loss']
        final_loss = progression[-1]['loss']
        initial_accuracy = progression[0]['accuracy']
        final_accuracy = progression[-1]['accuracy']
        
        print(f"Loss improvement: {initial_loss:.6f} → {final_loss:.6f} ({(initial_loss-final_loss)/initial_loss*100:.1f}% reduction)")
        print(f"Accuracy improvement: {initial_accuracy:.1f}% → {final_accuracy:.1f}% (+{final_accuracy-initial_accuracy:.1f}%)")
        print(f"Colors learned: {progression[0]['colors_learned']} → {progression[-1]['colors_learned']} (+{progression[-1]['colors_learned']-progression[0]['colors_learned']})")

def compare_training_runs(runs_info):
    """Compare multiple training runs
    
    Args:
        runs_info: List of dicts with 'checkpoint_dir' and 'session_id' keys
    """
    print(f"\n=== Training Run Comparison ===")
    print(f"{'Run':<20} {'Final Epoch':<12} {'Final Loss':<12} {'Final Acc':<12} {'Colors':<8} {'Time':<10}")
    print(f"{'='*20} {'='*12} {'='*12} {'='*12} {'='*8} {'='*10}")
    
    all_runs = []
    for i, run_info in enumerate(runs_info):
        progression = load_training_progression(run_info['checkpoint_dir'], run_info['session_id'])
        if progression:
            final_data = progression[-1]
            run_name = f"Run {i+1}"
            if 'name' in run_info:
                run_name = run_info['name']
            
            print(f"{run_name:<20} {final_data['epoch']:<12} {final_data['loss']:<12.6f} {final_data['accuracy']:<12.1f} {final_data['colors_learned']:<8} {final_data['elapsed_time']:<10.1f}")
            all_runs.append({
                'name': run_name,
                'data': final_data,
                'progression': progression
            })
    
    # Find best run
    if all_runs:
        best_run = max(all_runs, key=lambda x: x['data']['colors_learned'])
        print(f"\nBest run: {best_run['name']} with {best_run['data']['colors_learned']} colors learned")
    
    return all_runs

def analyze_convergence_patterns(checkpoint_dir="memorycapacityruns", session_id=None):
    """Analyze when and how the model converges"""
    progression = load_training_progression(checkpoint_dir, session_id)
    
    if not progression or len(progression) < 2:
        print("Insufficient data for convergence analysis.")
        return
    
    print(f"\n=== Convergence Analysis ===")
    
    # Calculate loss improvement rates
    loss_improvements = []
    acc_improvements = []
    
    for i in range(1, len(progression)):
        prev_loss = progression[i-1]['loss']
        curr_loss = progression[i]['loss']
        loss_improvement = (prev_loss - curr_loss) / prev_loss * 100
        loss_improvements.append(loss_improvement)
        
        prev_acc = progression[i-1]['accuracy']
        curr_acc = progression[i]['accuracy']
        acc_improvement = curr_acc - prev_acc
        acc_improvements.append(acc_improvement)
    
    # Identify convergence patterns
    recent_improvements = loss_improvements[-3:] if len(loss_improvements) >= 3 else loss_improvements
    avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)
    
    print(f"Average loss improvement (last 3 checkpoints): {avg_recent_improvement:.4f}%")
    
    if avg_recent_improvement < 0.01:
        print("⚠️  Model appears to have converged (minimal loss improvement)")
    elif avg_recent_improvement < 0.1:
        print("🐌 Model is converging slowly")
    else:
        print("📈 Model is still actively learning")
    
    # Show improvement timeline
    print(f"\n{'Checkpoint':<12} {'Loss Δ%':<12} {'Acc Δ':<12}")
    print(f"{'='*12} {'='*12} {'='*12}")
    
    for i, (loss_imp, acc_imp) in enumerate(zip(loss_improvements, acc_improvements)):
        checkpoint_num = i + 2  # Start from 2nd checkpoint
        print(f"{checkpoint_num:<12} {loss_imp:<12.4f} {acc_imp:<12.1f}")
    
    return {
        'loss_improvements': loss_improvements,
        'accuracy_improvements': acc_improvements,
        'converged': avg_recent_improvement < 0.01
    } 