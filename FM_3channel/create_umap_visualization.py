#!/usr/bin/env python3
"""
Script to create UMAP visualizations of bottleneck latent spaces from trained Flow Matching models.
Extracts 100 random points from each model's latent space and visualizes them using UMAP.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import pytorch_lightning as pl
from module import FlowMatchingModel
import warnings
warnings.filterwarnings('ignore')

class BottleneckExtractor:
    """Class to extract bottleneck representations from trained models"""
    
    def __init__(self, checkpoint_path):
        """Initialize with a trained model checkpoint"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.model = FlowMatchingModel.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Hook to capture bottleneck representations
        self.bottleneck_features = []
        self.model.scalar_field.bottleneck.register_forward_hook(self._hook_fn)
        
    def _hook_fn(self, module, input, output):
        """Hook function to capture bottleneck output"""
        # Global average pooling to get a single vector per sample
        pooled = torch.mean(output, dim=[2, 3])  # Average over spatial dimensions
        self.bottleneck_features.append(pooled.detach().cpu())
    
    def extract_features(self, total_mass_maps, star_maps, gas_maps, astro_params):
        """Extract bottleneck features from the given data"""
        # Clear previous batch features
        self.bottleneck_features = []
        
        # Data is already sampled and normalized at loading level
        total_mass = total_mass_maps
        star_maps = star_maps
        gas_maps = gas_maps
        astro_params = astro_params
        
        # Convert to tensors and move to device
        # IMPORTANT: Follow exact same pattern as training step
        total_mass = torch.FloatTensor(total_mass).unsqueeze(1).to(self.device)  # Shape: (batch, 1, H, W)
        target_maps = torch.stack([
            torch.FloatTensor(star_maps),
            torch.FloatTensor(gas_maps)
        ], dim=1).to(self.device)  # Shape: (batch, 2, H, W) - star and gas channels
        astro_params = torch.FloatTensor(astro_params).to(self.device)
        
        # Create dummy time and condition tensors for forward pass
        # Follow exact same pattern as training step
        batch_size = total_mass.size(0)
        t = torch.rand(batch_size, device=self.device)
        
        # Create x_t exactly as in training: interpolate between x0 and x1
        x0 = total_mass.expand(-1, 2, -1, -1)  # Expand to 2 channels like in training
        x1 = target_maps  # This is the target maps (star + gas)
        
        # Interpolate between x0 and x1 (same as training)
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Create condition_param exactly as in training: [t, astro_params]
        # Ensure astro_params has the correct batch size by repeating if necessary
        if astro_params.size(0) != batch_size:
            # Repeat astro_params to match batch size (this handles the //15 indexing)
            repeat_factor = (batch_size + astro_params.size(0) - 1) // astro_params.size(0)
            astro_params_expanded = astro_params.repeat(repeat_factor, 1)
            astro_params_expanded = astro_params_expanded[:batch_size]  # Trim to exact batch size
        else:
            astro_params_expanded = astro_params
        
        condition_param = torch.cat([t.unsqueeze(1).float(), astro_params_expanded.float()], dim=1)
        
        # Forward pass to trigger the hook - use EXACT same call pattern as training
        with torch.no_grad():
            # Debug: print tensor shapes to verify they match training step
            print(f"      Debug - x_t shape: {x_t.shape}")
            print(f"      Debug - condition_param shape: {condition_param.shape}")
            print(f"      Debug - total_mass shape: {total_mass.shape}")
            print(f"      Debug - target_maps shape: {target_maps.shape}")
            
            _ = self.model.scalar_field(x_t, condition_param, total_mass, target_maps)
        
        # Concatenate all captured features
        if self.bottleneck_features:
            features = torch.cat(self.bottleneck_features, dim=0)
            return features.numpy()
        else:
            return np.array([])
    
    def clear_memory(self):
        """Clear GPU memory and reset features"""
        self.bottleneck_features = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_data_for_model(model_name, data_dir="/n/netscratch/iaifi_lab/Lab/msliu/CMD/data", n_samples=100):
    """Load data for a specific cosmological model - only load n_samples instead of full dataset"""
    print(f"Loading data for {model_name}...")
    
    try:
        # Load the full datasets first to get the total size and compute normalization stats
        total_mass = np.load(f'{data_dir}/{model_name}/Maps_Mtot_{model_name}_LH_z=0.00.npy')
        star_maps = np.load(f'{data_dir}/{model_name}/Maps_Mstar_{model_name}_LH_z=0.00.npy')
        gas_maps = np.load(f'{data_dir}/{model_name}/Maps_Mgas_{model_name}_LH_z=0.00.npy')
        astro_params = np.loadtxt(f'{data_dir}/{model_name}/params_LH_{model_name}.txt')
        
        print(f"  Full dataset size: {len(total_mass)} samples")
        
        # IMPORTANT: Follow exact training normalization pattern
        # 1. Apply log1p transformation to full dataset first (same as training)
        total_mass_log = np.log1p(total_mass)
        star_maps_log = np.log1p(star_maps)
        gas_maps_log = np.log1p(gas_maps)
        
        # 2. Compute normalization stats from full log1p dataset (same as training)
        tot_mean, tot_std = total_mass_log.mean(), total_mass_log.std()
        star_mean, star_std = star_maps_log.mean(), star_maps_log.std()
        gas_mean, gas_std = gas_maps_log.mean(), gas_maps_log.std()
        
        print(f"  Normalization stats from full log1p dataset:")
        print(f"    Total mass: mean={tot_mean:.4f}, std={tot_std:.4f}")
        print(f"    Star maps: mean={star_mean:.4f}, std={star_std:.4f}")
        print(f"    Gas maps: mean={gas_mean:.4f}, std={gas_std:.4f}")
        
        # 3. Randomly sample n_samples from the log1p-transformed dataset
        if len(total_mass_log) > n_samples:
            indices = np.random.choice(len(total_mass_log), n_samples, replace=False)
            total_mass = total_mass_log[indices]  # Already log1p transformed
            star_maps = star_maps_log[indices]    # Already log1p transformed
            gas_maps = gas_maps_log[indices]      # Already log1p transformed
            astro_params = astro_params[indices//15]  # Same indexing as in training
            print(f"  Randomly sampled {n_samples} log1p-transformed samples from {model_name}")
        else:
            print(f"  Dataset smaller than requested samples, using all {len(total_mass_log)} samples")
            total_mass = total_mass_log  # Already log1p transformed
            star_maps = star_maps_log    # Already log1p transformed
            gas_maps = gas_maps_log      # Already log1p transformed
        
        # 4. Apply normalization to the sampled log1p data (same as training)
        total_mass = (total_mass - tot_mean) / tot_std
        star_maps = (star_maps - star_mean) / star_std
        gas_maps = (gas_maps - gas_mean) / gas_std
        
        astro_params = astro_params[:, :2]  # Only first 2 parameters
        
        print(f"  Final loaded: {len(total_mass)} normalized samples from {model_name}")
        return total_mass, star_maps, gas_maps, astro_params
        
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return None, None, None, None

def create_umap_visualization():
    """Main function to create UMAP visualization"""
    
    print("=" * 80)
    print("UMAP VISUALIZATION OF BOTTLENECK LATENT SPACES")
    print("=" * 80)
    print("This script will:")
    print("- Load 1000 samples from each cosmological model (using batch processing)")
    print("- Extract bottleneck representations from the newest trained model")
    print("- Create UMAP visualizations with multiple perspectives")
    print("- Process 4 cosmological models: IllustrisTNG, EAGLE, SIMBA, Astrid")
    print(f"- Total samples to process: 4 models × 1000 samples = 4000 samples")
    print("=" * 80)
    
    # Only use the newest model (oyc4eh50)
    models_config = {
        'Newest Model (oyc4eh50)': 'lightning_logs/oyc4eh50/checkpoints/best-model-epoch=88-val_loss=0.023915.ckpt'
    }
    
    # Define the cosmological models to analyze
    cosmological_models = ['IllustrisTNG', 'EAGLE', 'SIMBA', 'Astrid']
    
    # Number of samples to extract from each cosmological model
    n_samples_per_model = 5000
    
    # Batch size for processing (adjust based on your GPU memory)
    batch_size = 100  # Process 100 samples at a time
    
    # Store all features and labels for UMAP
    all_features = []
    all_labels = []
    all_model_names = []
    
    # Process the trained model
    for model_name, checkpoint_path in models_config.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
            
        try:
            # Load the trained model
            extractor = BottleneckExtractor(checkpoint_path)
            
            # Process each cosmological model
            for cosmo_model in cosmological_models:
                print(f"\nExtracting features from {cosmo_model} using {model_name}...")
                
                # Load data - only load n_samples instead of full dataset
                total_mass, star_maps, gas_maps, astro_params = load_data_for_model(
                    cosmo_model, n_samples=n_samples_per_model
                )
                
                if total_mass is None:
                    continue
                
                # Process data in batches to manage memory
                print(f"  Processing {len(total_mass)} samples in batches of {batch_size}...")
                
                batch_features = []
                total_batches = (len(total_mass) + batch_size - 1) // batch_size
                
                for i in range(0, len(total_mass), batch_size):
                    end_idx = min(i + batch_size, len(total_mass))
                    batch_start = i
                    batch_end = end_idx
                    current_batch = i // batch_size + 1
                    
                    print(f"    Processing batch {current_batch}/{total_batches} (samples {batch_start}-{batch_end-1})")
                    
                    # Extract bottleneck features for this batch
                    # Handle astro_params indexing correctly (same as training: indices//15)
                    astro_start = batch_start // 15
                    astro_end = (batch_end + 14) // 15  # Round up to include all needed params
                    
                    # Ensure we don't go out of bounds
                    astro_start = max(0, astro_start)
                    astro_end = min(len(astro_params), astro_end)
                    
                    print(f"      Using astro_params indices {astro_start}:{astro_end} for batch {batch_start}:{batch_end}")
                    
                    features = extractor.extract_features(
                        total_mass[batch_start:batch_end], 
                        star_maps[batch_start:batch_end], 
                        gas_maps[batch_start:batch_end], 
                        astro_params[astro_start:astro_end]  # Correctly indexed astro_params
                    )
                    
                    if len(features) > 0:
                        batch_features.append(features)
                        print(f"      Extracted {len(features)} features from batch")
                    else:
                        print(f"      No features extracted from batch")
                    
                    # Clear GPU memory between batches
                    extractor.clear_memory()
                    
                    # Progress update
                    print(f"      Progress: {current_batch}/{total_batches} batches completed")
                
                # Combine all batch features
                if batch_features:
                    # Stack all batch features together
                    combined_features = np.vstack(batch_features)
                    all_features.append(combined_features)
                    
                    # Create labels for the correct number of features
                    total_features_this_model = combined_features.shape[0]
                    all_labels.extend([f"{model_name} - {cosmo_model}"] * total_features_this_model)
                    all_model_names.extend([cosmo_model] * total_features_this_model)
                    
                    print(f"  Total extracted: {total_features_this_model} features from {cosmo_model}")
                    print(f"  Batch breakdown: {[len(batch) for batch in batch_features]} features per batch")
                else:
                    print(f"  No features extracted from {cosmo_model}")
                
                # Clear memory after processing each cosmological model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"  GPU memory cleared after processing {cosmo_model}")
                    
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_features:
        print("No features extracted. Exiting.")
        return
    
    # Combine all features
    print(f"\n{'='*60}")
    print("Creating UMAP visualization...")
    print(f"{'='*60}")
    
    # Concatenate all features
    X = np.vstack(all_features)
    print(f"Total features: {X.shape}")
    
    # Debug: Check that labels match features
    print(f"Total labels: {len(all_labels)}")
    print(f"Total model names: {len(all_model_names)}")
    print(f"Feature count: {X.shape[0]}")
    
    if len(all_labels) != X.shape[0]:
        print("ERROR: Label count doesn't match feature count!")
        print("This will cause the visualization to fail.")
        return
    
    # Data is already properly normalized using training normalization
    # No need for additional StandardScaler - use data as-is
    print("Using training-normalized features (no additional scaling needed)")
    
    # Create UMAP embedding with improved parameters
    print("Fitting UMAP with improved parameters...")
    reducer = umap.UMAP(
        n_neighbors=16,        # Increased from 15 for better local structure
        min_dist=0.1,         # Decreased from 0.1 for more spread
        n_components=2,
        random_state=42,
        metric='euclidean',
        spread=1.0,            # Controls how spread out the embedding is
        local_connectivity=1.0, # Better local structure preservation
        repulsion_strength=0.5  # Stronger repulsion between points
    )
    
    embedding = reducer.fit_transform(X)
    
    # If the embedding is still too linear, try alternative approach
    print("Checking embedding spread...")
    x_range = embedding[:, 0].max() - embedding[:, 0].min()
    y_range = embedding[:, 1].max() - embedding[:, 1].min()
    aspect_ratio = x_range / y_range if y_range > 0 else float('inf')
    
    if aspect_ratio > 5 or aspect_ratio < 0.2:  # Too linear
        print("Embedding appears too linear. Trying alternative UMAP approach...")
        # Try with different parameters for more spread
        reducer_alt = umap.UMAP(
            n_neighbors=50,        # Much higher for global structure
            min_dist=0.01,         # Much lower for maximum spread
            n_components=2,
            random_state=42,
            metric='cosine',       # Try cosine distance
            spread=1.5,            # Higher spread
            local_connectivity=2.0, # Higher local connectivity
            repulsion_strength=2.0  # Much stronger repulsion
        )
        embedding = reducer_alt.fit_transform(X)
        print("Applied alternative UMAP parameters for better spread")
    
    print(f"Final embedding shape: {embedding.shape}")
    print(f"Embedding ranges - X: [{embedding[:, 0].min():.3f}, {embedding[:, 0].max():.3f}], Y: [{embedding[:, 1].min():.3f}, {embedding[:, 1].max():.3f}]")
    
    # Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(12, 10))
    
    # Create single UMAP plot colored by cosmological model
    unique_cosmo_models = list(set(all_model_names))
    
    # Use much more distinct and high-contrast colors
    distinct_colors = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6']  # Red, Blue, Orange, Purple
    # Alternative even more distinct colors:
    # distinct_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF']  # Pure RGB + Magenta
    
    for i, cosmo_model in enumerate(unique_cosmo_models):
        mask = [name == cosmo_model for name in all_model_names]
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   c=[distinct_colors[i]], label=cosmo_model, 
                   alpha=0.9, s=10, linewidth=1.0)
    
    plt.title('Bottleneck Latent Space by Simulation Model', fontsize=15)
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)
    
    # Add legend with larger, more visible text
    plt.legend(fontsize=14, frameon=True, 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add sample count information
    sample_counts = {}
    for cosmo_model in unique_cosmo_models:
        count = sum(1 for name in all_model_names if name == cosmo_model)
        sample_counts[cosmo_model] = count
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "bottleneck_umap_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Also save the embedding data for further analysis
    np.savez("bottleneck_umap_data.npz", 
             embedding=embedding, 
             features=X, 
             labels=all_labels, 
             model_names=all_model_names)
    print("Embedding data saved to: bottleneck_umap_data.npz")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(embedding)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"UMAP embedding dimension: {embedding.shape[1]}")
    
    print(f"\nSamples per cosmological model:")
    for cosmo_model in unique_cosmo_models:
        count = sum(1 for name in all_model_names if name == cosmo_model)
        print(f"  {cosmo_model}: {count}")
    
    print(f"\nSamples per trained model:")
    print(f"  Newest Model (oyc4eh50): {len(embedding)} samples")

if __name__ == "__main__":
    create_umap_visualization()
