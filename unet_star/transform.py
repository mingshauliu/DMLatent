import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import kornia.filters as KF

class Augmentation(nn.Module):
    """Simple resize transform for validation/test sets (no augmentation)"""
    
    def __init__(self, size=(256, 256), scaling_scheme='log', normalize=False, blur_sigma=0.1, blur_kernel=0):
        super().__init__()
        self.size = size
        self.scaling_scheme = scaling_scheme
        self.normalize = normalize
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
            
    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        # Resize
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)  # [1, 1, H, W]
        img = torch.clamp(img, min=0)

        # === Apply constant Gaussian blur ===
        sigma = self.blur_sigma if hasattr(self, 'blur_sigma') else 0.0  # Use provided value or fallback
        if self.blur_kernel > 0 and sigma > 0:
            sigma_tensor = torch.tensor([[sigma, sigma]], device=img.device, dtype=img.dtype)
            img = KF.gaussian_blur2d(img, (self.blur_kernel, self.blur_kernel), sigma=sigma_tensor)

        # === Scaling ===
        if self.scaling_scheme == 'log':
            img = torch.log1p(img)
        elif self.scaling_scheme == 'tanh':
            img = torch.tanh(img)
        elif self.scaling_scheme == 'none':
            pass

        # === Normalize ===
        if self.normalize:
            img = (img - img.mean()) / img.std()

        return img.squeeze(0)  # Back to [1, H, W] or [H, W]
