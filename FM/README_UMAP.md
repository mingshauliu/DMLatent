# UMAP Visualization of Bottleneck Latent Spaces

This script creates UMAP visualizations of the bottleneck latent representations from the newest trained Flow Matching model. It extracts 100 random points from each cosmological model's latent space and visualizes them using UMAP dimensionality reduction.

## Overview

The script analyzes the newest trained model:
- **Newest Model (oyc4eh50)**: Best model from epoch 88 with validation loss 0.023915

For this trained model, it processes data from four cosmological models:
- IllustrisTNG
- EAGLE  
- SIMBA
- Astrid

## Features

- **Bottleneck Feature Extraction**: Uses PyTorch hooks to capture the bottleneck layer representations (512-dimensional vectors)
- **Random Sampling**: Extracts 100 random samples from each cosmological model
- **UMAP Visualization**: Creates 2D embeddings using UMAP with multiple visualization perspectives
- **Comprehensive Analysis**: Shows separation by cosmological model and detailed breakdown
- **ResNet Integration**: Properly handles the ResNet branch input which processes the target maps (star and gas channels)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_umap.txt
```

## Usage

Run the script from the FM/ directory:

```bash
cd FM/
python create_umap_visualization.py
```

## Output

The script generates:

1. **bottleneck_umap_visualization.png**: A 4-panel visualization showing:
   - All models combined
   - Separation by cosmological model
   - Separation by trained model
   - Detailed breakdown

2. **bottleneck_umap_data.npz**: Saved data containing:
   - `embedding`: 2D UMAP coordinates
   - `features`: Original bottleneck features (512D)
   - `labels`: Labels for each sample
   - `model_names`: Cosmological model names

## Technical Details

- **Bottleneck Layer**: The script extracts features from the bottleneck layer of the UNet architecture, which has 512 channels (base_channels*8 = 64*8)
- **Feature Processing**: Global average pooling is applied to convert spatial features to vectors
- **Data Preprocessing**: Applies the same log1p transformation used during training
- **ResNet Branch**: The target maps (star and gas channels) are properly processed through the ResNet branch as intended in the model architecture
- **UMAP Parameters**: Uses 15 neighbors, 0.1 minimum distance, and Euclidean metric

## Expected Results

The UMAP visualization should reveal:
- Clustering patterns between different cosmological models
- How the trained model encodes different astrophysical regimes in the latent space
- Potential separation of different physical conditions in the bottleneck representations

## Troubleshooting

- **Memory Issues**: The script loads large datasets. Ensure sufficient RAM/VRAM
- **Data Path**: Verify the data directory path in `load_data_for_model()`
- **Checkpoint Path**: Ensure the checkpoint file exists in the specified path
- **CUDA**: The script automatically detects and uses GPU if available
- **Tensor Dimensions**: The script properly handles tensor shapes for the UNet and ResNet inputs

## Customization

You can modify:
- Number of samples per model (currently 100)
- UMAP parameters (n_neighbors, min_dist, metric)
- Visualization layout and colors
- Data loading paths and preprocessing
