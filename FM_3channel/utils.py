import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

class AstroMapDataset(Dataset):
    def __init__(self, total_mass_maps: np.ndarray, star_maps: np.ndarray, gas_maps: np.ndarray, T_maps: np.ndarray, P_maps: np.ndarray, params: np.ndarray, model_ids: np.ndarray = None, transform=None):
        self.total_mass_maps = torch.FloatTensor(total_mass_maps)
        
        # Stack star and gas maps into 2-channel target
        star_tensor = torch.FloatTensor(star_maps)
        gas_tensor = torch.FloatTensor(gas_maps)
        T_tensor = torch.FloatTensor(T_maps)
        P_tensor = torch.FloatTensor(P_maps)
        self.target_maps = torch.stack([star_tensor, gas_tensor, T_tensor, P_tensor], dim=1)  # Shape: (N, 3, H, W)
        self.params = torch.FloatTensor(params)
        self.model_ids = model_ids  # Optional: track which model each sample comes from
        self.transform = transform
        
        # Normalize
        tot_mean, tot_std = total_mass_maps.mean(), total_mass_maps.std()
        star_mean, star_std = star_maps.mean(), star_maps.std()
        gas_mean, gas_std = gas_maps.mean(), gas_maps.std()
        T_mean, T_std = T_maps.mean(), T_maps.std()
        P_mean, P_std = P_maps.mean(), P_maps.std()
        print(f"Normalising tot log maps, mean: {tot_mean}, std: {tot_std}")
        print(f"Normalising star log maps, mean: {star_mean}, std: {star_std}")
        print(f"Normalising gas log maps, mean: {gas_mean}, std: {gas_std}")
        print(f"Normalising T log maps, mean: {T_mean}, std: {T_std}")
        print(f"Normalising P log maps, mean: {P_mean}, std: {P_std}")
        self.total_mass_maps = (self.total_mass_maps - tot_mean) / tot_std
        self.target_maps[:, 0] = (self.target_maps[:, 0] - star_mean) / star_std  # star channel
        self.target_maps[:, 1] = (self.target_maps[:, 1] - gas_mean) / gas_std    # gas channel
        self.target_maps[:, 2] = (self.target_maps[:, 2] - T_mean) / T_std    # T channel
        self.target_maps[:, 3] = (self.target_maps[:, 3] - P_mean) / P_std    # T channel
    def __len__(self):
        return len(self.total_mass_maps)
    
    def __getitem__(self, idx):
        total_mass = self.total_mass_maps[idx].unsqueeze(0)  # Shape: (1, H, W)
        target_map = self.target_maps[idx]  # Shape: (2, H, W)
        astro_param = self.params[idx] # Shape: (1, 6)
        
        if self.transform is not None:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            total_mass = self.transform(total_mass)
            torch.manual_seed(seed)
            target_map = self.transform(target_map)
        
        if self.model_ids is not None:
            model_id = self.model_ids[idx]
            return total_mass, target_map, astro_param, model_id
        else:
            return total_mass, target_map, astro_param