import sys
import os
import numpy as np
import math
import random

    # --- Main Backward Pass for Output Layer (now uses Leaky ReLU derivative and correct gradient) ---
class Layer:
    def __init__(self, input_size, output_size, is_output_layer=False, initial_weights=None, initial_biases=None, alpha=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha 

        if initial_weights is not None:
            self.weights = initial_weights
        else:
            # He initialization for ReLU-like activations
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            
        if initial_biases is not None:
            self.biases = initial_biases
        else:
            self.biases = np.zeros((1, output_size))
        
        self.input_data = None
        self.weighted_sum = None
        self.is_output_layer = is_output_layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        s = self.sigmoid(self.weighted_sum)
        return s * (1 - s)

    def leaky_relu(self, x):
        return np.maximum(self.alpha * x, x)
    
    def leaky_relu_derivative(self):
        deriv = np.ones_like(self.weighted_sum)
        deriv[self.weighted_sum <= 0] = self.alpha
        return deriv

    def forward(self, input_data):
        self.input_data = input_data
        self.weighted_sum = np.dot(input_data, self.weights) + self.biases
        
        if np.any(np.isnan(self.weighted_sum)) or np.any(np.isinf(self.weighted_sum)):
            # print(f"WARNING: NaN/Inf detected in weighted_sum of layer with output size {self.output_size}. Input data shape: {input_data.shape}, Weights shape: {self.weights.shape}")
            return np.full_like(self.weighted_sum, np.nan)

        return self.leaky_relu(self.weighted_sum)

    def backward(self, incoming_gradient, learning_rate, grad_clip_norm):
        if np.any(np.isnan(incoming_gradient)) or np.any(np.isinf(incoming_gradient)):
            # print(f"WARNING: NaN/Inf detected in incoming_gradient of layer with output size {self.output_size}. Skipping update.")
            return np.full_like(self.input_data, np.nan)

        s_derivative = self.leaky_relu_derivative()
        w_sum_derivative = s_derivative * incoming_gradient
        
        d_weights = np.dot(self.input_data.T, w_sum_derivative)
        d_biases = np.sum(w_sum_derivative, axis=0, keepdims=True)
        
        d_weights = np.clip(d_weights, -grad_clip_norm, grad_clip_norm)
        d_biases = np.clip(d_biases, -grad_clip_norm, grad_clip_norm)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return np.dot(w_sum_derivative, self.weights.T)

    def backward_output_layer(self, target_vector, final_result, learning_rate, grad_clip_norm):
        if np.any(np.isnan(final_result)) or np.any(np.isinf(final_result)):
            # print(f"WARNING: NaN/Inf detected in final_result of output layer. Skipping update.")
            return np.full_like(self.input_data, np.nan)

        loss_gradient = 2 * (final_result - target_vector)
        
        s_derivative = self.leaky_relu_derivative()
        w_sum_derivative = s_derivative * loss_gradient
        
        d_weights = np.dot(self.input_data.T, w_sum_derivative)
        d_biases = np.sum(w_sum_derivative, axis=0, keepdims=True)

        d_weights = np.clip(d_weights, -grad_clip_norm, grad_clip_norm)
        d_biases = np.clip(d_biases, -grad_clip_norm, grad_clip_norm)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return np.dot(w_sum_derivative, self.weights.T)