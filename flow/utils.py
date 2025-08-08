import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

class AstroMapDataset(Dataset):
    """Dataset for paired total mass and star maps"""
    
    def __init__(self, total_mass_maps: np.ndarray, star_maps: np.ndarray, transform=None):
    
        self.total_mass_maps = torch.FloatTensor(total_mass_maps)
        self.star_maps = torch.FloatTensor(star_maps)
        self.transform = transform
        
        tot_mean,tot_std = total_mass_maps.mean(), total_mass_maps.std()
        star_mean,star_std = star_maps.mean(), star_maps.std()
    
        print(f'Normalising with tot_log mean: {tot_mean:.3f}, std: {tot_std:.3f}')
        print(f'Normalising with star_log mean: {star_mean:.3f}, std: {star_std:.3f}')

        self.total_mass_maps = (self.total_mass_maps - tot_mean) / tot_std
        self.star_maps = (self.star_maps - star_mean) / star_std
            
    def __len__(self):
        return len(self.total_mass_maps)
    
    def __getitem__(self, idx):
        total_mass = self.total_mass_maps[idx].unsqueeze(0)  # Add channel dim
        star_map = self.star_maps[idx].unsqueeze(0)
        
        if self.transform is not None:
            # Apply same transform to both maps
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            total_mass = self.transform(total_mass)
            torch.manual_seed(seed)
            star_map = self.transform(star_map)
        
        return total_mass, star_map


