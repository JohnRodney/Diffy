import sys
import os
import numpy as np
import math
import random
import datetime

from neuralnet.layer import Layer
from neuralnet.text_autoencoder import TextAutoencoder, create_text_autoencoder
from tokenizer.tokenizer import Tokenizer
from utils.drift_analysis import calculate_vector_drift
from utils.checkpoint_manager import save_checkpoint
from utils.model_persistence import save_model, load_model

# Backward compatibility wrapper
class Network(TextAutoencoder):
    """Backward compatibility wrapper for the old Network class"""
    
    def __init__(self, vector_length, hidden_layer_count, bottleneck_size, alpha=0.01):
        config = {
            'vector_length': vector_length,
            'hidden_layer_count': hidden_layer_count,
            'bottleneck_size': bottleneck_size,
            'activation_type': 'leaky_relu',
            'activation_alpha': alpha
        }
        super().__init__(config)

# Keep the old Network class structure but use the new architecture internally
class LegacyNetwork:
    def __init__(self, vector_length, hidden_layer_count, bottleneck_size, alpha=0.01):
        self.vector_length = vector_length
        self.hidden_layer_count = hidden_layer_count
        self.bottleneck_size = bottleneck_size
        self.alpha = alpha

        self.all_layers = []
        self._build_network()

    def _build_network(self):
        current_hidden_layer_size = self.vector_length
        reduction_factor = math.ceil((self.vector_length - self.bottleneck_size) / self.hidden_layer_count)

        # Encoder Layers
        self.encoder_layers = []
        self.encoder_layers.append(Layer(self.vector_length, current_hidden_layer_size, alpha=self.alpha)) # Input Layer
        for _ in range(self.hidden_layer_count):
            last_hidden_layer_size = current_hidden_layer_size
            next_hidden_layer_size = current_hidden_layer_size - reduction_factor
            if next_hidden_layer_size < self.bottleneck_size:
                next_hidden_layer_size = self.bottleneck_size
            current_hidden_layer_size = next_hidden_layer_size
            self.encoder_layers.append(Layer(last_hidden_layer_size, current_hidden_layer_size, alpha=self.alpha))
        
        # Decoder Layers
        self.decoder_layers = []
        self.decoder_layers.append(Layer(current_hidden_layer_size, current_hidden_layer_size, alpha=self.alpha)) # Bottleneck layer (input to decoder)
        for _ in range(self.hidden_layer_count):
            prev_layer_size = current_hidden_layer_size
            next_hidden_layer_size = current_hidden_layer_size + reduction_factor
            if next_hidden_layer_size > self.vector_length:
                next_hidden_layer_size = self.vector_length
            current_hidden_layer_size = next_hidden_layer_size
            self.decoder_layers.append(Layer(prev_layer_size, current_hidden_layer_size, alpha=self.alpha))
        
        self.decoder_output_layer = Layer(current_hidden_layer_size, self.vector_length, is_output_layer=True, alpha=self.alpha)
        self.decoder_layers.append(self.decoder_output_layer)

        self.all_layers = self.encoder_layers + self.decoder_layers

    def forward(self, input_data):
        current_output = input_data
        for layer in self.all_layers:
            current_output = layer.forward(current_output)
            if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                return np.full_like(current_output, np.nan) # Propagate NaN/Inf
        return current_output

    def backward(self, target_batch, final_reconstruction, learning_rate, grad_clip_norm):
        # Start backward pass from the output layer
        incoming_gradient = self.decoder_output_layer.backward_output_layer(target_batch, final_reconstruction, learning_rate, grad_clip_norm)
        
        if np.any(np.isnan(incoming_gradient)) or np.any(np.isinf(incoming_gradient)):
            return np.nan # Indicate NaN/Inf in backward pass

        # Propagate through remaining layers in reverse
        # We iterate through all_layers[:-1] because decoder_output_layer was handled separately
        for layer in reversed(self.all_layers[:-1]):
            incoming_gradient = layer.backward(incoming_gradient, learning_rate, grad_clip_norm)
            if np.any(np.isnan(incoming_gradient)) or np.any(np.isinf(incoming_gradient)):
                return np.nan # Indicate NaN/Inf in backward pass
        return incoming_gradient # This return value is typically discarded after the first layer

    def encode(self, input_data):
        current_output = input_data
        for layer in self.encoder_layers:
            current_output = layer.forward(current_output)
            if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                return np.full_like(current_output, np.nan)
        return current_output

    def infer(self, input_word, model_filename, words, vector_length):
        tokenizer = Tokenizer(2)
        tokenizer.fill_dictionary(words)
        vector_dictionary = {}
        for word, index in tokenizer.vocab.items():
            vector_dictionary[word] = np.random.rand(1, vector_length)

        input_vector = vector_dictionary[input_word]
        reconstructed_output = self.forward(input_vector)
        return reconstructed_output



# Removed duplicate functions - now using utilities from utils/ directory

def get_batches(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def run_training(
    vector_length,
    words,
    hidden_layer_count,
    bottleneck_size,
    num_epochs,
    learning_rate,
    batch_size,
    grad_clip_norm,
    leaky_relu_alpha,
    quick_test_epochs=0, # New parameter for quick testing
    should_load_model=False,
    checkpoint_interval=500, # Save checkpoint every N epochs
    checkpoint_dir="memorycapacityruns" # Directory to save checkpoints
):
    tokenizer = Tokenizer(2)
    tokenizer.fill_dictionary(words)
    vector_dictionary = {}
    for word, index in tokenizer.vocab.items():
        vector_dictionary[word] = np.random.rand(1, vector_length)

    # Initialize the network
    network = Network(vector_length, hidden_layer_count, bottleneck_size, alpha=leaky_relu_alpha)
    
    # Generate unique session ID based on timestamp
    session_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    start_time = datetime.datetime.now()
    print(f"Training session ID: {session_id}")
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Bundle all training parameters for metadata logging
    training_params = {
        "vector_length": vector_length,
        "words": words,
        "hidden_layer_count": hidden_layer_count,
        "bottleneck_size": bottleneck_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_clip_norm": grad_clip_norm,
        "leaky_relu_alpha": leaky_relu_alpha,
        "checkpoint_interval": checkpoint_interval
    }
    
    model_filename = "my_autoencoder_model.npz"

   
    file_exists = os.path.exists(model_filename)
    if file_exists and should_load_model:
        print("Attempting to load existing model...")
        if load_model(network, model_filename):
            print("Model loaded successfully. Continuing training or testing.")
        else:
            print("Failed to load model. Starting training from scratch.")
    elif file_exists and not should_load_model:
        model_filename = f"my_autoencoder_model{random.randint(0, 1000000)}.npz"
    else:
        print("No existing model found. Starting training from scratch.")

    training_data_pairs = [(vector_dictionary[word], vector_dictionary[word]) for word in words]
    
    # Save initial vectors for drift analysis
    initial_vectors = {}
    for word in words:
        initial_vectors[word] = vector_dictionary[word].copy()
    
    word_show_counts = {word: 0 for word in words}

    print("Starting training...")
    print(f"Initial vectors saved for drift analysis ({len(initial_vectors)} colors)")


    # --- Quick Test Phase ---
    if quick_test_epochs > 0:
        print(f"\n--- Running Quick Test for {quick_test_epochs} Epochs ---")
        for epoch in range(quick_test_epochs):
            total_epoch_loss = 0
            batches = list(get_batches(training_data_pairs, batch_size))
            
            for batch in batches:
                input_batch = np.vstack([pair[0] for pair in batch])
                target_batch = np.vstack([pair[1] for pair in batch])

                current_output = network.forward(input_batch)
                if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                    total_epoch_loss = np.nan
                    break

                sample_loss = np.mean((target_batch - current_output)**2)
                total_epoch_loss += sample_loss

                # No backward pass during quick test, just observe forward loss
            
            avg_epoch_loss = total_epoch_loss / len(training_data_pairs)
            
            # Only print every 10 epochs during quick test
            if (epoch + 1) % 10 == 0:
                print(f"Quick Test Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.8f}")
            
            if np.isnan(avg_epoch_loss) or avg_epoch_loss > 1e10: # Check for early explosion in test
                print("Quick test indicates instability (NaN/inf loss). Adjust parameters before full training.")
                return # Exit if quick test fails
        print("--- Quick Test Complete. Proceeding to Full Training ---")

    

    # --- Full Training Phase ---
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        
        batches = list(get_batches(training_data_pairs, batch_size))

        for batch in batches:
            input_batch = np.vstack([pair[0] for pair in batch])
            target_batch = np.vstack([pair[1] for pair in batch])

            for pair in batch:
                for word, vec in vector_dictionary.items():
                    if np.array_equal(vec, pair[0]):
                        word_show_counts[word] += 1
                        break

            current_output = network.forward(input_batch)
            if np.any(np.isnan(current_output)) or np.any(np.isinf(current_output)):
                print(f"Stopping batch due to NaN/Inf in forward pass at epoch {epoch+1}.")
                total_epoch_loss = np.nan
                break
            
            final_reconstruction = current_output

            sample_loss = np.mean((target_batch - final_reconstruction)**2)
            total_epoch_loss += sample_loss

            # Perform backward pass for the batch
            backward_status = network.backward(target_batch, final_reconstruction, learning_rate, grad_clip_norm)
            if np.any(np.isnan(backward_status)) or np.any(np.isinf(backward_status)):
                print(f"Stopping batch due to NaN/Inf in backward pass at epoch {epoch+1}.")
                total_epoch_loss = np.nan
                break

        avg_epoch_loss = total_epoch_loss / len(training_data_pairs)
        
        if np.isnan(avg_epoch_loss):
            print(f"Training stopped due to NaN loss at Epoch {epoch+1}. Consider restarting with smaller learning rate or more aggressive clipping.")
            break

        # Print full reconstruction every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"\n--- Reconstruction Test at Epoch {epoch+1} ---")
            print_reconstruction_compact(words, vector_dictionary, network, tokenizer, word_show_counts, epoch)
            print("-" * 80)

        # Save checkpoint at specified intervals
        if (epoch + 1) % checkpoint_interval == 0:
            # Calculate current accuracy for checkpoint
            correct_count = 0
            for word_to_test in words:
                input_vec = vector_dictionary[word_to_test]
                reconstructed_output = network.forward(input_vec)
                
                if not (np.any(np.isnan(reconstructed_output)) or np.any(np.isinf(reconstructed_output))):
                    best_match = None
                    best_match_similarity = -2.0
                    for word_in_vocab, vec_in_vocab_idx in tokenizer.vocab.items():
                        vec_in_vocab = vector_dictionary[list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(vec_in_vocab_idx)]]
                        
                        dot_product = np.dot(reconstructed_output.flatten(), vec_in_vocab.flatten())
                        magnitude_result = np.linalg.norm(reconstructed_output.flatten())
                        magnitude_word = np.linalg.norm(vec_in_vocab.flatten())
                        
                        if magnitude_result != 0 and magnitude_word != 0:
                            cosine_similarity = dot_product / (magnitude_result * magnitude_word)
                            if cosine_similarity > best_match_similarity:
                                best_match_similarity = cosine_similarity
                                best_match = word_in_vocab
                    
                    if best_match == word_to_test:
                        correct_count += 1
            
            accuracy = (correct_count / len(words)) * 100
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Calculate vector drift data
            drift_data = calculate_vector_drift(network, words, vector_dictionary, initial_vectors, tokenizer)
            
            save_checkpoint(network, epoch + 1, avg_epoch_loss, accuracy, checkpoint_dir, session_id, training_params, start_time, elapsed_time, drift_data)

        if avg_epoch_loss < 1e-8:
            print(f"\nConverged at Epoch {epoch+1}")
            break
    
    save_model(network, model_filename)

    print("\n--- Training Complete. Testing Reconstruction ---")
    print_reconstruction_compact(words, vector_dictionary, network, tokenizer, word_show_counts, epoch)


def print_reconstruction_compact(words, vector_dictionary, network, tokenizer, word_show_counts, epoch):
    """Compact version showing all color reconstructions in a readable format"""
    correct_count = 0
    results = []
    
    for word_to_test in words:
        input_vec = vector_dictionary[word_to_test]
        reconstructed_output = network.forward(input_vec)
        
        if np.any(np.isnan(reconstructed_output)) or np.any(np.isinf(reconstructed_output)):
            best_match = "NaN/Inf"
            best_match_similarity = -2.0
            reconstruction_error = np.nan
        else:
            best_match = None
            best_match_similarity = -2.0
            for word_in_vocab, vec_in_vocab_idx in tokenizer.vocab.items():
                vec_in_vocab = vector_dictionary[list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(vec_in_vocab_idx)]]

                dot_product = np.dot(reconstructed_output.flatten(), vec_in_vocab.flatten())
                magnitude_result = np.linalg.norm(reconstructed_output.flatten())
                magnitude_word = np.linalg.norm(vec_in_vocab.flatten())
                
                if magnitude_result == 0 or magnitude_word == 0:
                    cosine_similarity = -1.0
                else:
                    cosine_similarity = dot_product / (magnitude_result * magnitude_word)

                if cosine_similarity > best_match_similarity:
                    best_match_similarity = cosine_similarity
                    best_match = word_in_vocab
            
            reconstruction_error = np.mean((vector_dictionary[word_to_test] - reconstructed_output)**2)
        
        # Track correct reconstructions
        if best_match == word_to_test:
            correct_count += 1
            
        results.append(f"{word_to_test} → {best_match}")
    
    # Print summary
    accuracy = (correct_count / len(words)) * 100
    print(f"Accuracy: {correct_count}/{len(words)} ({accuracy:.1f}%)")
    
    # Print results in columns
    print("\nReconstruction Results:")
    for i in range(0, len(results), 4):  # 4 columns
        row = results[i:i+4]
        print("  ".join(f"{r:<20}" for r in row))
    
    print(f"\nCorrect: {correct_count}, Confused: {len(words) - correct_count} epochs: {epoch}")

