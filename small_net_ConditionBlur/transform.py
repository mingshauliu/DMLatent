import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import random

class LogTransform:
    def __call__(self, tensor):
        return torch.log1p(tensor)

class TensorAugment(nn.Module):
    """Data augmentation module for tensors"""
    
    def __init__(self, size=(256, 256), p_flip=0.5, p_rot=0.5,
                 noise_std=0.01, apply_log=True, blur_kernel=5,
                 blur_sigma_range=(0.1, 1.5), normalize=False, blur_sigma=None):
        super().__init__()
        self.size = size
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.noise_std = noise_std
        self.apply_log = apply_log
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range
        self.normalize = normalize
        self.blur_sigma = blur_sigma

    def forward(self, img):  # img: [1, H, W] or [H, W]
        if img.dim() == 2:
            img = img.unsqueeze(0)  # [1, H, W]
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]  # Take only first channel

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)  # [1, 1, H, W]

        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[3])  # H-flip
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[2])  # V-flip
        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            img = torch.rot90(img, k=k, dims=[2, 3])

        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise

        # === Kornia Blur ===
        sigma = 0.0
        if self.blur_kernel > 0:
            sigma = random.uniform(0,self.blur_sigma_range) if self.blur_sigma is None else self.blur_sigma
            if sigma > 0:
                sigma_tensor = torch.tensor([[sigma, sigma]], device=img.device)
                img = KF.gaussian_blur2d(img, (self.blur_kernel, self.blur_kernel), sigma=sigma_tensor)

        img = img.squeeze(0)  # [1, H, W]

        if self.apply_log:
            img = torch.clamp(img, min=0)
            img = torch.log1p(img)

        if self.normalize:
            img = (img - img.mean()) / img.std()

        return img, sigma

class ResizeBlur(nn.Module):
    """Simple resize transform for validation/test sets (with optional blur)"""
    
    def __init__(self, size=(256, 256), apply_log=True, normalize=False,
                 blur_kernel=0, blur_sigma_range=None):
        
        super().__init__()
        self.size = size
        self.apply_log = apply_log
        self.normalize = normalize
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range

    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)  # [1, 1, H, W]

        # Determine sigma
        sigma = 0.0
        if self.blur_kernel > 0:
            if self.blur_sigma_range is not None:
                sigma = random.uniform(0.1,self.blur_sigma_range)

            if sigma > 0:
                sigma_tensor = torch.tensor([[sigma, sigma]], device=img.device)
                img = KF.gaussian_blur2d(img, (self.blur_kernel, self.blur_kernel), sigma=sigma_tensor)

        img = img.squeeze(0)  # [1, H, W]

        if self.apply_log:
            img = torch.clamp(img, min=0)
            img = torch.log1p(img)

        if self.normalize:
            img = (img - img.mean()) / img.std()

        return img, sigma
