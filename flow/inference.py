#!/usr/bin/env python3
"""
Star Map Generation Inference Script

This script loads a trained flow matching model and generates star maps from total mass maps.
Supports both single images and batch processing.

Usage:
    python inference.py --model_path model.ckpt --input total_mass.npy --output star_map.npy
    python inference.py --model_path model.ckpt --input total_mass.png --output star_map.png
    python inference.py --model_path model.ckpt --input_dir mass_maps/ --output_dir star_maps/
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl
from pathlib import Path
import json

# Import the model class (assuming it's in the same directory or installed)
try:
    from test_flow import FlowMatchingModel  
except ImportError:
    print("Please ensure the FlowMatchingModel class is available")
    print("Either put this script in the same directory as your model definition")
    print("or adjust the import path")
    exit(1)


class StarMapGenerator:
    """Inference class for generating star maps from total mass maps"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the generator with a trained model
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> FlowMatchingModel:
        """Load the trained model from checkpoint"""
        print(f"Loading model from {model_path}")
        model = FlowMatchingModel.load_from_checkpoint(model_path, map_location=self.device)
        model.to(self.device)
        return model
    
    def _normalize_map(self, map_array: np.ndarray) -> np.ndarray:
        """Normalize a map to [0, 1] range"""
        map_min = map_array.min()
        map_max = map_array.max()
        if map_max - map_min == 0:
            return np.zeros_like(map_array)
        return (map_array - map_min) / (map_max - map_min)
    
    def _denormalize_map(self, normalized_map: np.ndarray, 
                        original_min: float, original_max: float) -> np.ndarray:
        """Denormalize a map back to original range"""
        return normalized_map * (original_max - original_min) + original_min
    
    def _preprocess_input(self, total_mass_map: np.ndarray) -> torch.Tensor:
        """
        Preprocess input for model inference
        
        Args:
            total_mass_map: Input array of shape (H, W) or (1, H, W)
            
        Returns:
            Preprocessed tensor of shape (1, 1, H, W)
        """
        # Ensure 2D
        if total_mass_map.ndim == 3 and total_mass_map.shape[0] == 1:
            total_mass_map = total_mass_map[0]
        elif total_mass_map.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {total_mass_map.shape}")
        
        # Normalize
        normalized = self._normalize_map(total_mass_map)
        
        # Convert to tensor and add batch + channel dimensions
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return tensor.to(self.device)
    
    def _postprocess_output(self, output_tensor: torch.Tensor, 
                          target_min: float = None, target_max: float = None) -> np.ndarray:
        """
        Postprocess model output to numpy array
        
        Args:
            output_tensor: Model output of shape (1, 1, H, W)
            target_min: Target minimum value for denormalization
            target_max: Target maximum value for denormalization
            
        Returns:
            Numpy array of shape (H, W)
        """
        # Convert to numpy and remove batch + channel dimensions
        output_np = output_tensor.cpu().detach().numpy()[0, 0]  # (H, W)
        
        # Ensure values are in [0, 1] range
        output_np = np.clip(output_np, 0, 1)
        
        # Denormalize if target range provided
        if target_min is not None and target_max is not None:
            output_np = self._denormalize_map(output_np, target_min, target_max)
        
        return output_np
    
    def generate_star_map(self, total_mass_map: np.ndarray, 
                         num_steps: int = 100, 
                         preserve_range: bool = True) -> np.ndarray:
        """
        Generate a star map from a total mass map
        
        Args:
            total_mass_map: Input total mass map as numpy array (H, W)
            num_steps: Number of ODE integration steps
            preserve_range: Whether to preserve the original value range
            
        Returns:
            Generated star map as numpy array (H, W)
        """
        # Store original range for denormalization
        original_min = total_mass_map.min() if preserve_range else None
        original_max = total_mass_map.max() if preserve_range else None
        
        # Preprocess input
        input_tensor = self._preprocess_input(total_mass_map)
        
        # Generate star map
        with torch.no_grad():
            output_tensor = self.model.sample(
                total_mass_condition=input_tensor,
                num_steps=num_steps,
                method='euler'
            )
        
        # Postprocess output
        star_map = self._postprocess_output(
            output_tensor, 
            target_min=original_min, 
            target_max=original_max
        )
        
        return star_map
    
    def generate_batch(self, total_mass_maps: np.ndarray, 
                      num_steps: int = 100) -> np.ndarray:
        """
        Generate star maps for a batch of total mass maps
        
        Args:
            total_mass_maps: Input maps of shape (N, H, W)
            num_steps: Number of ODE integration steps
            
        Returns:
            Generated star maps of shape (N, H, W)
        """
        batch_size = total_mass_maps.shape[0]
        
        # Preprocess all maps
        input_tensors = []
        original_ranges = []
        
        for i in range(batch_size):
            mass_map = total_mass_maps[i]
            original_ranges.append((mass_map.min(), mass_map.max()))
            input_tensors.append(self._preprocess_input(mass_map))
        
        # Stack into batch
        batch_tensor = torch.cat(input_tensors, dim=0)  # (N, 1, H, W)
        
        # Generate star maps
        with torch.no_grad():
            output_batch = self.model.sample(
                total_mass_condition=batch_tensor,
                num_steps=num_steps,
                method='euler'
            )
        
        # Postprocess all outputs
        star_maps = []
        for i in range(batch_size):
            star_map = self._postprocess_output(
                output_batch[i:i+1],  # Keep batch dimension
                target_min=original_ranges[i][0],
                target_max=original_ranges[i][1]
            )
            star_maps.append(star_map)
        
        return np.array(star_maps)


def load_image_file(file_path: str) -> np.ndarray:
    """Load an image file as numpy array"""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        # Load as grayscale
        img = Image.open(file_path).convert('L')
        return np.array(img, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def save_image_file(array: np.ndarray, file_path: str, colormap: str = 'viridis'):
    """Save numpy array as image file"""
    if file_path.endswith('.npy'):
        np.save(file_path, array)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        # Normalize to [0, 255] and save as image
        normalized = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        Image.fromarray(normalized).save(file_path)
    elif file_path.endswith('.pdf'):
        # Save as matplotlib plot
        plt.figure(figsize=(10, 8))
        plt.imshow(array, cmap=colormap, origin='lower')
        plt.colorbar(label='Intensity')
        plt.title('Generated Star Map')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        raise ValueError(f"Unsupported output format: {file_path}")


def create_comparison_plot(total_mass: np.ndarray, star_map: np.ndarray, 
                          output_path: str, colormap: str = 'viridis'):
    """Create a side-by-side comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Total mass map
    im1 = ax1.imshow(total_mass, cmap=colormap, origin='lower')
    ax1.set_title('Total Mass Map (Input)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Mass Density')
    
    # Generated star map
    im2 = ax2.imshow(star_map, cmap=colormap, origin='lower')
    ax2.set_title('Generated Star Map (Output)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Star Density')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate star maps from total mass maps')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', help='Input total mass map file')
    parser.add_argument('--output', help='Output star map file')
    parser.add_argument('--input_dir', help='Directory of input total mass maps')
    parser.add_argument('--output_dir', help='Directory for output star maps')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of ODE integration steps')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--colormap', default='viridis', help='Colormap for visualization')
    parser.add_argument('--comparison', action='store_true', help='Create comparison plots')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for directory processing')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = StarMapGenerator(args.model_path, args.device)
    print(f"Model loaded successfully on {generator.device}")
    
    if args.input and args.output:
        # Single file processing
        print(f"Processing single file: {args.input}")
        
        # Load input
        total_mass_map = load_image_file(args.input)
        print(f"Input shape: {total_mass_map.shape}")
        
        # Generate star map
        print("Generating star map...")
        star_map = generator.generate_star_map(
            total_mass_map, 
            num_steps=args.num_steps
        )
        
        # Save output
        save_image_file(star_map, args.output, args.colormap)
        print(f"Star map saved to: {args.output}")
        
        # Create comparison plot if requested
        if args.comparison:
            comparison_path = args.output.replace('.', '_comparison.')
            if not comparison_path.endswith('.pdf'):
                comparison_path = comparison_path.rsplit('.', 1)[0] + '_comparison.pdf'
            create_comparison_plot(total_mass_map, star_map, comparison_path, args.colormap)
            print(f"Comparison plot saved to: {comparison_path}")
    
    elif args.input_dir and args.output_dir:
        # Directory processing
        print(f"Processing directory: {args.input_dir}")
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all input files
        input_files = list(input_dir.glob('*.npy')) + list(input_dir.glob('*.png')) + \
                     list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg'))
        
        print(f"Found {len(input_files)} input files")
        
        # Process in batches
        for i in range(0, len(input_files), args.batch_size):
            batch_files = input_files[i:i+args.batch_size]
            print(f"Processing batch {i//args.batch_size + 1}/{(len(input_files)-1)//args.batch_size + 1}")
            
            # Load batch
            batch_maps = []
            for file_path in batch_files:
                total_mass_map = load_image_file(str(file_path))
                batch_maps.append(total_mass_map)
            
            # Ensure all maps have the same shape
            if len(set(map.shape for map in batch_maps)) > 1:
                print("Warning: Maps have different shapes, processing individually")
                for j, (file_path, mass_map) in enumerate(zip(batch_files, batch_maps)):
                    star_map = generator.generate_star_map(mass_map, args.num_steps)
                    output_path = output_dir / f"{file_path.stem}_star{file_path.suffix}"
                    save_image_file(star_map, str(output_path), args.colormap)
            else:
                # Batch processing
                batch_array = np.array(batch_maps)
                star_maps = generator.generate_batch(batch_array, args.num_steps)
                
                # Save results
                for file_path, star_map in zip(batch_files, star_maps):
                    output_path = output_dir / f"{file_path.stem}_star{file_path.suffix}"
                    save_image_file(star_map, str(output_path), args.colormap)
                    
                    if args.comparison:
                        comparison_path = output_dir / f"{file_path.stem}_comparison.pdf"
                        mass_map = load_image_file(str(file_path))
                        create_comparison_plot(mass_map, star_map, str(comparison_path), args.colormap)
        
        print(f"All files processed. Results saved to: {output_dir}")
    
    else:
        print("Error: Must specify either (--input and --output) or (--input_dir and --output_dir)")
        parser.print_help()


if __name__ == "__main__":
    main()