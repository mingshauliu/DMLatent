import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import random
from scipy.ndimage import gaussian_filter
import numpy as np
# def to_fourier(tensor, sigma):
#     """Apply Gaussian blur to tensor"""
#     if tensor.dim() == 3:  # [C, H, W]
#         tensor = tensor.unsqueeze(0)  # [1, C, H, W]
#     return gaussian_filter(tensor,sigma=sigma)
    
# def gaussian_blur_tensor(tensor, kernel_size=5, sigma=1.0):
#     """Apply Gaussian blur to tensor"""
#     if tensor.dim() == 3:  # [C, H, W]
#         tensor = tensor.unsqueeze(0)  # [1, C, H, W]
#     return gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)

def blur(tensor, sigma):
    """Apply Gaussian blur to tensor and return as PyTorch tensor"""

    # Ensure input is 4D: [B, C, H, W]
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # Move to CPU and convert to numpy
    tensor_np = tensor.cpu().numpy()

    # Apply Gaussian filter to each channel independently
    blurred_np = np.empty_like(tensor_np)
    for i in range(tensor_np.shape[1]):  # channel-wise blur
        for b in range(tensor_np.shape[0]):
            blurred_np[b, i] = gaussian_filter(tensor_np[b, i], sigma=sigma)

    # Convert back to torch tensor, match original dtype and device
    blurred_tensor = torch.from_numpy(blurred_np).to(tensor.device).type(tensor.dtype)

    return blurred_tensor

class LogTransform:
    def __call__(self, tensor):
        return torch.log1p(tensor)

class TensorAugment(nn.Module):
    """Data augmentation module for tensors"""
    
    def __init__(self, size=(256, 256), p_flip=0.5, p_rot=0.5,
                 noise_std=0.01, apply_log=True, blur_kernel=5, blur_sigma=1.0, normalize=None):
        super().__init__()
        self.size = size
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.noise_std = noise_std
        self.apply_log = apply_log
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.normalize = normalize

    def forward(self, img):  # img: [1, H, W] or [H, W]
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        # Resize to target size
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        img = img.squeeze(0)  # [1, H, W]

        # Random flips and rotation
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[2])  # H-flip
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[1])  # V-flip
        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            img = torch.rot90(img, k=k, dims=[1, 2])

        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise

        # Optional blur
        if self.blur_kernel>0:
            img = blur(img, sigma=self.blur_sigma).squeeze(0)

        # Optional log scale
        if self.apply_log:
            img = torch.clamp(img, min=0)
            img = torch.log1p(img)
            
        if self.normalize is not None:
            img = (img - self.normalize['mean']) / self.normalize['std'] # z-score normalisation 

        return img


class SimpleResize(nn.Module):
    """Simple resize transform for validation/test sets (no augmentation)"""
    
    def __init__(self, size=(256, 256), apply_log=True, normalize=None):
        super().__init__()
        self.size = size
        self.apply_log = apply_log
        self.normalize = normalize

    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        if self.apply_log:
            img = torch.clamp(img, min=0)
            img = torch.log1p(img)
            
        if self.normalize is not None:
            img = (img - self.normalize['mean']) / self.normalize['std'] # z-score normalisation 
            
        return img.squeeze(0)