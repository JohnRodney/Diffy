import numpy as np

def compute_pca_weights(training_data, bottleneck_size):
    """
    Compute PCA-based encoder and decoder weights using pure NumPy
    
    training_data: (n_samples, input_dim) - training vectors
    bottleneck_size: int - size of compressed representation
    
    Returns:
        encoder_weights: (input_dim, bottleneck_size) 
        decoder_weights: (bottleneck_size, input_dim)
    """
    # Center the data
    mean = np.mean(training_data, axis=0)
    centered_data = training_data - mean
    
    # SVD to get principal components
    U, S, Vt = np.linalg.svd(centered_data.T, full_matrices=False)
    
    # Take first bottleneck_size components
    # Vt shape: (input_dim, n_samples), we want first bottleneck_size rows
    pca_components = Vt[:bottleneck_size]  # Shape: (bottleneck_size, input_dim)
    
    # Encoder: transpose to (input_dim, bottleneck_size)
    encoder_weights = pca_components.T
    
    # Decoder: keep as (bottleneck_size, input_dim) 
    decoder_weights = pca_components
    
    return encoder_weights.astype(np.float32), decoder_weights.astype(np.float32) 