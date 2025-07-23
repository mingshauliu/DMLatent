import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleResize(nn.Module):
    """Simple resize transform for validation/test sets (no augmentation)"""
    
    def __init__(self, size=(256, 256), scaling_scheme='log', normalize=False):
        super().__init__()
        self.size = size
        self.scaling_scheme = scaling_scheme
        self.normalize = normalize

    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        img = torch.clamp(img, min=0)

        if self.scaling_scheme == 'log':
            img = torch.log1p(img)

        elif self.scaling_scheme == 'tanh':
            img = torch.tanh(img)

        elif self.scaling_scheme == 'none':
            pass
            
        if self.normalize:
            img = (img - img.mean()) / img.std() # Normalize to zero mean and unit variance
            
        return img.squeeze(0)