#!/usr/bin/env python3
"""
Simple example of using the trained flow matching model to generate star maps

This is a minimal example showing how to:
1. Load a trained model
2. Process a single total mass map
3. Generate and visualize the result
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from test_flow import FlowMatchingModel  # Adjust import as needed


def generate_star_map_simple(model_path, total_mass_map, num_steps=100, device='auto'):
    """
    Simple function to generate a star map from a total mass map
    
    Args:
        model_path: Path to trained model checkpoint (.ckpt file)
        total_mass_map: numpy array of shape (H, W) containing the total mass map
        num_steps: Number of integration steps for the flow
        device: 'auto', 'cpu', or 'cuda'
    
    Returns:
        Generated star map as numpy array of shape (H, W)
    """
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained model
    print(f"Loading model from {model_path}")
    model = FlowMatchingModel.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    # Preprocess input
    # Normalize to [0, 1]
    mass_min, mass_max = total_mass_map.min(), total_mass_map.max()
    normalized_mass = (total_mass_map - mass_min) / (mass_max - mass_min + 1e-8)
    
    # Convert to tensor: (1, 1, H, W)
    input_tensor = torch.FloatTensor(normalized_mass).unsqueeze(0).unsqueeze(0).to(device)
    
    # Generate star map
    print("Generating star map...")
    with torch.no_grad():
        output_tensor = model.sample(
            total_mass_condition=input_tensor,
            num_steps=num_steps,
            method='euler'
        )
    
    # Postprocess output
    star_map = output_tensor.cpu().numpy()[0, 0]  # Remove batch and channel dims
    star_map = np.clip(star_map, 0, 1)  # Ensure valid range
    
    # Denormalize to original range (optional)
    star_map = star_map * (mass_max - mass_min) + mass_min
    
    return star_map


def visualize_comparison(total_mass_map, star_map, save_path=None):
    """Create a side-by-side comparison visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot total mass map
    im1 = ax1.imshow(total_mass_map, cmap='viridis', origin='lower')
    ax1.set_title('Total Mass Map (Input)', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Mass Density')
    
    # Plot generated star map
    im2 = ax2.imshow(star_map, cmap='viridis', origin='lower')
    ax2.set_title('Generated Star Map (Output)', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Star Density')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example with dummy data (replace with your actual data)
    
    # Path to your trained model
    MODEL_PATH = "path/to/your/trained_model.ckpt"
    
    # Example 1: Generate from dummy data
    print("Example 1: Generating from synthetic data")
    
    # Create a synthetic total mass map (replace with your actual data loading)
    height, width = 128, 128
    x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    total_mass_map = np.exp(-(x**2 + y**2)) + 0.3 * np.random.rand(height, width)
    
    # Generate star map
    star_map = generate_star_map_simple(MODEL_PATH, total_mass_map, num_steps=50)
    
    # Visualize results
    visualize_comparison(total_mass_map, star_map, save_path="example_comparison.png")
    
    print("\n" + "="*50)
    
    # Example 2: Load from file
    print("Example 2: Loading from file")
    
    try:
        # Load your actual data
        total_mass_map = np.load("your_total_mass_map.npy")  # Replace with your file
        
        # Generate star map
        star_map = generate_star_map_simple(MODEL_PATH, total_mass_map, num_steps=100)
        
        # Save results
        np.save("generated_star_map.npy", star_map)
        visualize_comparison(total_mass_map, star_map, save_path="real_data_comparison.png")
        
        print("Star map generated and saved successfully!")
        
    except FileNotFoundError:
        print("File not found. Please update the file path or use the synthetic example above.")
    
    print("\n" + "="*50)
    
    # Example 3: Batch processing multiple maps
    print("Example 3: Batch processing")
    
    # Create multiple synthetic maps
    batch_size = 5
    total_mass_maps = []
    
    for i in range(batch_size):
        # Create different synthetic patterns
        sigma = 0.5 + 0.5 * i / batch_size
        mass_map = np.exp(-(x**2 + y**2) / sigma**2) + 0.2 * np.random.rand(height, width)
        total_mass_maps.append(mass_map)
    
    # Process each map
    star_maps = []
    for i, mass_map in enumerate(total_mass_maps):
        print(f"Processing map {i+1}/{batch_size}")
        star_map = generate_star_map_simple(MODEL_PATH, mass_map, num_steps=50)
        star_maps.append(star_map)
    
    # Create a grid visualization
    fig, axes = plt.subplots(2, batch_size, figsize=(3*batch_size, 6))
    
    for i in range(batch_size):
        # Top row: total mass maps
        axes[0, i].imshow(total_mass_maps[i], cmap='viridis', origin='lower')
        axes[0, i].set_title(f'Mass Map {i+1}')
        axes[0, i].axis('off')
        
        # Bottom row: generated star maps
        axes[1, i].imshow(star_maps[i], cmap='viridis', origin='lower')
        axes[1, i].set_title(f'Star Map {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("batch_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Batch processing complete!")


# Quick function for jupyter notebook usage
def quick_generate(model_path, input_array, steps=100):
    """One-liner function for jupyter notebooks"""
    return generate_star_map_simple(model_path, input_array, num_steps=steps)