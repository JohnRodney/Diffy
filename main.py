#!/usr/bin/env python3
"""
Main entry point for the noise generation project.
Loads data and saves the first image locally.
"""

from data_loader import DataLoader
from neuralnet.network import run_training, Network
from color_loader import get_color_names
import os
import argparse

def main():
    """Main entry point for the application."""
    print("Starting noise generation project...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the noise generation project.")
    parser.add_argument("-f", "--filename", type=str, help="The filename of the model to load.")
    parser.add_argument("-l", "--load_model", action="store_true", help="Load the model from the file.")
    parser.add_argument("-i", "--infer", action="store_true", help="Infer the model from the file.")
    
    args = parser.parse_args()
    # Define your hyperparameters here
    # if -f filename and -l load_model, then load the model from the file -i tells us to infer instead

    params = {
        "vector_length": 128,
        "words": get_color_names(),
        "hidden_layer_count": 3,
        "bottleneck_size": 56,
        "num_epochs": 10000,
        "learning_rate": 0.00001, # Adjusted learning rate
        "batch_size": 2,
        "grad_clip_norm": 0.01,   # Adjusted gradient clipping
        "leaky_relu_alpha": 0.01,
        "quick_test_epochs": 100 # Run 100 epochs as a quick test before full training
    }


    # use args -i for the input word
    if args.infer:
        network = Network(params["vector_length"], params["hidden_layer_count"], params["bottleneck_size"], params["leaky_relu_alpha"])
        network.infer(args.input_word, args.filename, params["words"], params["vector_length"])
    else:
        run_training(**params)
