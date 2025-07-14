import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import random

def gaussian_blur_tensor(tensor, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to tensor"""
    if tensor.dim() == 3:  # [C, H, W]
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]
    return gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)

def fft2_magnitude(tensor):
    """Compute 2D FFT and return magnitude spectrum (log-scaled)"""
    # tensor: [1, H, W]
    fft = torch.fft.fft2(tensor)
    fft_shifted = torch.fft.fftshift(fft)  # Center zero-frequency component
    magnitude = torch.abs(fft_shifted)
    return magnitude

class TensorAugment(nn.Module):
    """Data augmentation module for tensors (Fourier input)"""

    def __init__(self, size=(256, 256), p_flip=0.5, p_rot=0.5,
                 noise_std=0.01, apply_log=True, blur_kernel=5, blur_sigma=1.0, normalize=False):
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

        # Resize
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)

        # Augmentations in spatial domain
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[2])  # H-flip
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[1])  # V-flip
        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            img = torch.rot90(img, k=k, dims=[1, 2])
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
        if self.blur_kernel > 0:
            img = gaussian_blur_tensor(img, kernel_size=self.blur_kernel, sigma=self.blur_sigma).squeeze(0)

        # Fourier transform (magnitude spectrum)
        img = fft2_magnitude(img)

        # Log scale
        if self.apply_log:
            img = torch.log1p(img)

        # Normalize
        if self.normalize:
            img = (img - img.mean()) / img.std()

        return img


class SimpleResize(nn.Module):
    """Resize + 2D FFT for validation/test sets"""

    def __init__(self, size=(256, 256), apply_log=True, normalize=False):
        super().__init__()
        self.size = size
        self.apply_log = apply_log
        self.normalize = normalize

    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)

        # Fourier transform (magnitude)
        img = fft2_magnitude(img)

        if self.apply_log:
            img = torch.log1p(img)

        if self.normalize:
            img = (img - img.mean()) / img.std()

        return img
