#!/usr/bin/env python3
"""
Vector drift analysis utility for tracking how color vectors evolve during training
"""

import numpy as np
import json
import os

def calculate_vector_drift(network, words, vector_dictionary, initial_vectors, tokenizer):
    """Calculate vector drift metrics for each color"""
    drift_data = {
        "color_analysis": {},
        "summary": {
            "total_colors": len(words),
            "locked_in_colors": [],
            "drifting_colors": [],
            "failed_colors": []
        }
    }
    
    for word in words:
        current_vector = vector_dictionary[word]
        initial_vector = initial_vectors[word]
        
        # Calculate drift distance (L2 norm)
        drift_distance = float(np.linalg.norm(current_vector - initial_vector))
        
        # Test reconstruction quality
        reconstructed = network.forward(current_vector)
        
        if np.any(np.isnan(reconstructed)) or np.any(np.isinf(reconstructed)):
            reconstruction_quality = 0.0
            best_match = "NaN/Inf"
            best_similarity = -2.0
        else:
            # Find best match
            best_match = None
            best_similarity = -2.0
            
            for candidate_word in words:
                candidate_vector = vector_dictionary[candidate_word]
                dot_product = np.dot(reconstructed.flatten(), candidate_vector.flatten())
                mag_reconstructed = np.linalg.norm(reconstructed.flatten())
                mag_candidate = np.linalg.norm(candidate_vector.flatten())
                
                if mag_reconstructed > 0 and mag_candidate > 0:
                    similarity = dot_product / (mag_reconstructed * mag_candidate)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = candidate_word
            
            reconstruction_quality = float(best_similarity)
        
        # Determine color status
        is_locked_in = (best_match == word)
        is_drifting = (drift_distance > 0.1 and not is_locked_in)  # Threshold for "significant drift"
        
        # Store detailed analysis
        drift_data["color_analysis"][word] = {
            "drift_distance": drift_distance,
            "reconstruction_quality": reconstruction_quality,
            "best_match": best_match,
            "best_similarity": float(best_similarity),
            "is_locked_in": is_locked_in,
            "is_drifting": is_drifting,
            "initial_vector_norm": float(np.linalg.norm(initial_vector)),
            "current_vector_norm": float(np.linalg.norm(current_vector))
        }
        
        # Update summary lists
        if is_locked_in:
            drift_data["summary"]["locked_in_colors"].append(word)
        elif is_drifting:
            drift_data["summary"]["drifting_colors"].append(word)
        else:
            drift_data["summary"]["failed_colors"].append(word)
    
    # Add summary statistics
    drift_data["summary"]["locked_in_count"] = len(drift_data["summary"]["locked_in_colors"])
    drift_data["summary"]["drifting_count"] = len(drift_data["summary"]["drifting_colors"])
    drift_data["summary"]["failed_count"] = len(drift_data["summary"]["failed_colors"])
    
    return drift_data

def analyze_vector_drift(checkpoint_dir="memorycapacityruns", session_id=None):
    """Analyze vector drift patterns across training checkpoints"""
    from utils.training_analysis import load_training_progression
    
    progression = load_training_progression(checkpoint_dir, session_id)
    
    if not progression:
        print("No training progression data found.")
        return
    
    print(f"\n=== Vector Drift Analysis ===")
    print(f"Session: {session_id}")
    print(f"Checkpoints: {len(progression)}")
    
    # Show drift progression over time
    print(f"\n{'Epoch':<8} {'Locked':<8} {'Drifting':<10} {'Failed':<8} {'Avg Drift':<12}")
    print(f"{'='*8} {'='*8} {'='*10} {'='*8} {'='*12}")
    
    for data in progression:
        if data['vector_drift']:
            drift = data['vector_drift']
            summary = drift['summary']
            
            # Calculate average drift distance
            total_drift = sum(color_data['drift_distance'] for color_data in drift['color_analysis'].values())
            avg_drift = total_drift / len(drift['color_analysis'])
            
            print(f"{data['epoch']:<8} {summary['locked_in_count']:<8} {summary['drifting_count']:<10} {summary['failed_count']:<8} {avg_drift:<12.4f}")
    
    # Analyze final checkpoint in detail
    if progression and progression[-1]['vector_drift']:
        final_drift = progression[-1]['vector_drift']
        print(f"\n=== Final Checkpoint Analysis ===")
        print(f"Locked in colors ({len(final_drift['summary']['locked_in_colors'])}): {', '.join(final_drift['summary']['locked_in_colors'])}")
        print(f"Drifting colors ({len(final_drift['summary']['drifting_colors'])}): {', '.join(final_drift['summary']['drifting_colors'])}")
        
        # Show most/least drifted colors
        color_drifts = [(color, data['drift_distance']) for color, data in final_drift['color_analysis'].items()]
        color_drifts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nMost drifted colors:")
        for color, drift_dist in color_drifts[:5]:
            status = "LOCKED" if final_drift['color_analysis'][color]['is_locked_in'] else "DRIFT"
            print(f"  {color}: {drift_dist:.4f} ({status})")
        
        print(f"\nLeast drifted colors:")
        for color, drift_dist in color_drifts[-5:]:
            status = "LOCKED" if final_drift['color_analysis'][color]['is_locked_in'] else "DRIFT"
            print(f"  {color}: {drift_dist:.4f} ({status})")
    
    return progression

def analyze_initial_seeds(checkpoint_dir="memorycapacityruns", session_id=None):
    """Analyze relationship between initial seeds and final success"""
    from utils.training_analysis import load_training_progression
    
    progression = load_training_progression(checkpoint_dir, session_id)
    
    if not progression or not progression[-1]['vector_drift']:
        print("No drift data available for seed analysis.")
        return
    
    final_drift = progression[-1]['vector_drift']
    
    # Analyze initial vector characteristics vs success
    locked_in_seeds = []
    failed_seeds = []
    
    for color, data in final_drift['color_analysis'].items():
        seed_info = {
            'color': color,
            'initial_norm': data['initial_vector_norm'],
            'final_drift': data['drift_distance'],
            'success': data['is_locked_in']
        }
        
        if data['is_locked_in']:
            locked_in_seeds.append(seed_info)
        else:
            failed_seeds.append(seed_info)
    
    print(f"\n=== Initial Seed Analysis ===")
    print(f"Successful colors: {len(locked_in_seeds)}")
    print(f"Failed colors: {len(failed_seeds)}")
    
    if locked_in_seeds and failed_seeds:
        locked_norms = [s['initial_norm'] for s in locked_in_seeds]
        failed_norms = [s['initial_norm'] for s in failed_seeds]
        
        print(f"\nInitial vector norms:")
        print(f"  Successful: avg={np.mean(locked_norms):.4f}, std={np.std(locked_norms):.4f}")
        print(f"  Failed: avg={np.mean(failed_norms):.4f}, std={np.std(failed_norms):.4f}")
        
        # Check if there's a significant difference
        if np.mean(locked_norms) > np.mean(failed_norms):
            print("  → Successful colors had higher initial norms on average")
        else:
            print("  → Failed colors had higher initial norms on average")
    
    return {'successful': locked_in_seeds, 'failed': failed_seeds} 