import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

class AstroMapDataset(Dataset):
    def __init__(self, total_mass_maps: np.ndarray, star_maps: np.ndarray, gas_maps: np.ndarray, transform=None):
        self.total_mass_maps = torch.FloatTensor(total_mass_maps)
        
        # Stack star and gas maps into 2-channel target
        star_tensor = torch.FloatTensor(star_maps)
        gas_tensor = torch.FloatTensor(gas_maps)
        self.target_maps = torch.stack([star_tensor, gas_tensor], dim=1)  # Shape: (N, 2, H, W)
        
        self.transform = transform
        
        # Normalize
        tot_mean, tot_std = total_mass_maps.mean(), total_mass_maps.std()
        star_mean, star_std = star_maps.mean(), star_maps.std()
        gas_mean, gas_std = gas_maps.mean(), gas_maps.std()
        
        print(f"Normalising tot log maps, mean: {tot_mean}, std: {tot_std}")
        print(f"Normalising star log maps, mean: {star_mean}, std: {star_std}")
        print(f"Normalising gas log maps, mean: {gas_mean}, std: {gas_std}")

        self.total_mass_maps = (self.total_mass_maps - tot_mean) / tot_std
        self.target_maps[:, 0] = (self.target_maps[:, 0] - star_mean) / star_std  # star channel
        self.target_maps[:, 1] = (self.target_maps[:, 1] - gas_mean) / gas_std    # gas channel

    def __len__(self):
        return len(self.total_mass_maps)
    
    def __getitem__(self, idx):
        total_mass = self.total_mass_maps[idx].unsqueeze(0)  # Shape: (1, H, W)
        target_map = self.target_maps[idx]  # Shape: (2, H, W)
        
        if self.transform is not None:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            total_mass = self.transform(total_mass)
            torch.manual_seed(seed)
            target_map = self.transform(target_map)
        
        return total_mass, target_map