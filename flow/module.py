import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from utils import AstroMapDataset
from models import UNetScalarField, CNNScalarField

class AstroFlowMatchingDataModule(pl.LightningDataModule):
    """flow matching data pairs"""
    
    def __init__(self, 
                 total_mass_maps: np.ndarray,
                 star_maps: np.ndarray,
                 batch_size: int = 32,
                 val_split: float = 0.2,
                 num_workers: int = 4
                ):
        super().__init__()
        self.total_mass_maps = total_mass_maps
        self.star_maps = star_maps
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        # Split data
        n_samples = len(self.total_mass_maps)
        n_val = int(n_samples * self.val_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_total_mass = self.total_mass_maps[train_indices]
        train_star_maps = self.star_maps[train_indices]
        val_total_mass = self.total_mass_maps[val_indices]
        val_star_maps = self.star_maps[val_indices]
        
        self.train_dataset = AstroMapDataset(train_total_mass, train_star_maps)
        self.val_dataset = AstroMapDataset(val_total_mass, val_star_maps)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

class FlowMatchingModel(pl.LightningModule):
    """Flow Matching model for transforming total mass maps to star maps"""
    
    def __init__(self, 
                 architecture='unet',  
                 noise_std = 0.0,
                 learning_rate=1e-4,
                 alpha = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.alpha = alpha
        
        if architecture == 'unet':
            self.scalar_field = UNetScalarField(in_channels=3) 
        elif architecture == 'cnn':
            self.scalar_field = CNNScalarField(in_channels=3)
        else:
            raise ValueError("Architecture must be 'unet' or 'cnn'")
    
    def sample_time(self, batch_size, device):
        """Sample random times for flow matching"""
        return torch.rand(batch_size, device=device)
    
    def forward(self, x, t, condition):
        """Forward pass through the scalar field network"""
        return self.scalar_field(x, t, condition)
    
    def training_step(self, batch, batch_idx):
        total_mass, star_map = batch
        batch_size = total_mass.size(0)
        device = total_mass.device

        noise = torch.randn_like(total_mass)*self.noise_std # Gaussian noise mean=0, std=0.5 by default
        
        t = self.sample_time(batch_size, device)
        
        x0 = total_mass + noise
        x1 = star_map
        
        # Interpolate between x0 and x1
        
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Compute target scalar field
        target_field = x1-x0
        
        # Predict scalar field
        predicted_field = self(x_t, t, total_mass)
        
        # Compute loss in real space
        loss = F.mse_loss(predicted_field, target_field)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        total_mass, star_map = batch
        batch_size = total_mass.size(0)
        device = total_mass.device
        
        # Sample random times
        t = self.sample_time(batch_size, device)
        
        noise = torch.randn_like(total_mass)*self.noise_std # Gaussian noise mean=0, std=0.5 by default

        x0 = total_mass + noise
        x1 = star_map
        
        # Interpolate
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Compute target and predicted fields
        target_field = x1-x0
        predicted_field = self(x_t, t, total_mass)
        
        # Compute loss
        loss = F.mse_loss(predicted_field, target_field)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
        
    def sample(self, total_mass, total_mass_condition, num_steps=100, method='euler'):
        """Generate star maps from total mass maps using the learned flow"""
        self.eval()
        device = next(self.parameters()).device
        batch_size = total_mass.size(0)
        
        # Initialize x at time t = 0
        x = total_mass.clone()  # ensure x and condition aren't shared
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((batch_size,), i * dt, device=device)  # t ∈ [0, 1)
                field_change = self(x, t, total_mass_condition)
    
                if method == 'euler':
                    x = x + dt * field_change
                else:
                    raise ValueError("Only 'euler' method implemented")
        
        return x

