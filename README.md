# Diffy

_A Neural Network Color Learning Experiment (Future Diffusion Model)_

## What It Does Right Now

Diffy is currently a **color name autoencoder** that learns to associate color names with high-dimensional vector representations. It's the foundation for what will become a diffusion model.

### Current Capabilities

- **Color Data Collection**: Scrapes 264 color names and hex values from Wikipedia's "List of colors: A–F" page
- **Neural Network Training**: Autoencoder with leaky ReLU activation and MSE loss
- **Color Learning**: Attempts to learn distinct 128-dimensional representations for each color name
- **Reconstruction Testing**: Shows which colors the network can distinguish vs. which ones it confuses

### Architecture

```mermaid
graph TD
    A["Color Names<br/>(264 colors)<br/>e.g. 'Absolute_Zero'"] --> B["Tokenizer<br/>Word → ID Mapping"]
    B --> C["Random Vectors<br/>128D per color"]

    C --> D["Input Layer<br/>128 dimensions"]
    D --> E["Hidden Layer 1<br/>~110 dimensions"]
    E --> F["Hidden Layer 2<br/>~83 dimensions"]
    F --> G["Hidden Layer 3<br/>~70 dimensions"]
    G --> H["BOTTLENECK<br/>56 dimensions<br/><strong>0.21D per color!</strong>"]

    H --> I["Hidden Layer 4<br/>~70 dimensions"]
    I --> J["Hidden Layer 5<br/>~83 dimensions"]
    J --> K["Hidden Layer 6<br/>~110 dimensions"]
    K --> L["Output Layer<br/>128 dimensions"]

    L --> M["Reconstruction<br/>Should match input"]
    M --> N["Best Match Search<br/>Cosine Similarity"]
    N --> O["Predicted Color Name"]

    subgraph "Encoder"
        D
        E
        F
        G
    end

    subgraph "Decoder"
        I
        J
        K
        L
    end

    style H fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff
    style A fill:#74c0fc,stroke:#333,stroke-width:2px
    style O fill:#51cf66,stroke:#333,stroke-width:2px
    style M fill:#ffd43b,stroke:#333,stroke-width:2px
```

**Key Architecture Details:**

- **Input**: Color names (264 colors from Wikipedia)
- **Encoder**: 128D → 3 hidden layers → 56D bottleneck
- **Decoder**: 56D → 3 hidden layers → 128D reconstruction
- **Challenge**: 0.21 dimensions per color in the bottleneck (testing memorization limits!)
- **Activation**: Leaky ReLU (α=0.01) throughout
- **Loss**: Mean Squared Error (MSE)

## Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Scrape color data** (optional - colors_a_f.json already included):

   ```bash
   python scrape_colors.py
   ```

3. **Train the color autoencoder**:
   ```bash
   python main.py
   ```

## Current Experiment Results

The network is attempting to learn 264 different color names with only 56 latent dimensions - that's ~0.2 dimensions per color! This tests the fundamental limits of neural network memorization capacity.

**Hypothesis**: Perfect separation requires ~1 dimension per item, so 56D for 264 colors should be challenging.

## Roadmap

### Phase 1: Multi-modal Autoencoder (Next)

- [ ] Add image encoding (CNN layers)
- [ ] Shared latent space for text+image
- [ ] GPU acceleration
- [ ] Color patch generation

### Phase 2: Diffusion Model (Future)

- [ ] Implement diffusion process
- [ ] Noise scheduling
- [ ] Stable diffusion architecture
- [ ] Text-to-color-image generation

### Phase 3: Full Pipeline (Goal)

- [ ] Text prompt → Color image generation
- [ ] Quality improvements
- [ ] Additional modalities

## Technical Details

- **Framework**: Pure NumPy implementation (no PyTorch/TensorFlow)
- **Architecture**: Autoencoder with leaky ReLU (α=0.01)
- **Training**: MSE loss, gradient clipping, careful hyperparameters
- **Data**: Wikipedia color names + random 128D vectors (testing pure memorization)

## Files

- `main.py` - Main training entry point
- `scrape_colors.py` - Wikipedia color data scraper
- `color_loader.py` - Loads and cleans color names from JSON
- `neuralnet/network.py` - Autoencoder implementation
- `neuralnet/gemini_layer.py` - Neural network layer with leaky ReLU
- `tokenizer/tokenizer.py` - Simple text tokenization
- `colors_a_f.json` - Scraped Wikipedia color data

## Current Status

**Working**: Color name autoencoder, data collection, training pipeline
**In Progress**: Testing memorization capacity with 264 colors  
**Next**: Multi-modal architecture for text + images
**Future**: Full diffusion model implementation

---

_Note: This is an experimental neural network project. The "diffusion" capabilities are planned but not yet implemented. Currently focuses on fundamental color representation learning._
