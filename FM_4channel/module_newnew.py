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
from models_newnew import UNetScalarField, ResNetBranch

class AstroFlowMatchingDataModule(pl.LightningDataModule):
    """flow matching data pairs"""
    
    def __init__(self, 
                 cdm_mass_maps: np.ndarray,
                 star_maps: np.ndarray,
                 gas_maps: np.ndarray,
                 T_maps: np.ndarray,
                 P_maps: np.ndarray,
                 astro_params: np.ndarray,
                 batch_size: int = 32,
                 val_split: float = 0.2,
                 num_workers: int = 4
                ):
        super().__init__()
        self.cdm_mass_maps = cdm_mass_maps
        self.star_maps = star_maps
        self.gas_maps = gas_maps
        self.T_maps = T_maps
        self.P_maps = P_maps
        self.astro_params = astro_params
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        # Split data
        n_samples = len(self.cdm_mass_maps)
        n_val = int(n_samples * self.val_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_cdm_mass = self.cdm_mass_maps[train_indices]
        train_star_maps = self.star_maps[train_indices]
        train_gas_maps = self.gas_maps[train_indices]
        train_T_maps = self.T_maps[train_indices]
        train_P_maps = self.P_maps[train_indices]
        train_astro_params = self.astro_params[train_indices]
        
        val_cdm_mass = self.cdm_mass_maps[val_indices]
        val_star_maps = self.star_maps[val_indices]
        val_gas_maps = self.gas_maps[val_indices]
        val_T_maps = self.T_maps[val_indices]
        val_P_maps = self.P_maps[val_indices]
        val_astro_params = self.astro_params[val_indices]
        
        # Create model_ids for tracking (optional)
        train_model_ids = np.arange(len(train_indices))  # Simple sequential IDs
        val_model_ids = np.arange(len(val_indices))
        
        self.train_dataset = AstroMapDataset(
            train_cdm_mass,
            train_star_maps,
            train_gas_maps,
            train_T_maps,
            train_P_maps,
            params=train_astro_params,
            model_ids=train_model_ids,
            transform=None,
        )
        self.val_dataset = AstroMapDataset(
            val_cdm_mass,
            val_star_maps,
            val_gas_maps,   
            val_T_maps,
            val_P_maps,
            params=val_astro_params,
            model_ids=val_model_ids,
            transform=None,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Re-enable pin_memory for faster GPU transfer
            persistent_workers=True,  # Re-enable persistent workers for efficiency
            drop_last=False  # Don't drop incomplete batches
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,  # Re-enable pin_memory for faster GPU transfer
            persistent_workers=True,  # Re-enable persistent workers for efficiency
            drop_last=False  # Don't drop incomplete batches
        )

class FlowMatchingModel(pl.LightningModule):
    """Flow Matching model for transforming total mass maps to star maps"""
    
    def __init__(self, 
                 architecture='unet',
                 noise_std=0.0,
                 learning_rate=1e-4,
                 alpha = 0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.noise_std = noise_std
        self.scalar_field = UNetScalarField(in_channels=4, out_channels=4)
        self.resnet_branch = ResNetBranch(in_channels=4, embedding_dim=8)
        # Enable memory efficient training
        self.automatic_optimization = True
        self.automatic_logging = True
            
    def sample_time(self, batch_size, device):
        """Sample random times for flow matching"""
        return torch.rand(batch_size, device=device)
    
    def forward(self, x, t, resnet_input):
        """Forward pass through the scalar field network"""
        return self.scalar_field(x, t, resnet_input)
    
    def training_step(self, batch, batch_idx):
        # Handle optional model_id
        if len(batch) == 4:
            cdm_mass, target_maps, params, model_id = batch
        else:
            cdm_mass, target_maps, params = batch
            model_id = None

        batch_size = cdm_mass.size(0)
        device = cdm_mass.device
        
        t = self.sample_time(batch_size, device)
        x0 = cdm_mass.expand(-1,4,-1,-1)
        
        noise = torch.randn_like(x0)*self.noise_std
        x0 = x0 + noise
        
        x1 = target_maps
        
        # Interpolate between x0 and x1
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Compute target scalar field
        target_field = x1-x0

        condition_param = torch.cat([t.unsqueeze(1).float(), params.float()], dim=1)  # Shape: (batch, 7)
        
        # Compute ResNet embedding here (maintains same training behavior)
        resnet_embed = self.resnet_branch(target_maps)  # Shape: (batch, 8)
        
        # Predict scalar field
        predicted_field = self(x_t, condition_param, resnet_embed)
        # Compute loss in real space
        loss = F.mse_loss(predicted_field, target_field)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Handle optional model_id
        if len(batch) == 4:
            cdm_mass, target_maps, params, model_id = batch
        else:
            cdm_mass, target_maps, params = batch
            model_id = None
            
        batch_size = cdm_mass.size(0)
        device = cdm_mass.device
        
        # Sample random times
        t = self.sample_time(batch_size, device)
        
        x0 = cdm_mass.expand(-1,4,-1,-1)

        noise = torch.randn_like(x0)*self.noise_std
        x0 = x0 + noise
        x1 = target_maps
        
        # Interpolate
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        condition_param = torch.cat([t.unsqueeze(1).float(), params.float()], dim=1)  # Shape: (batch, 7)
        
        # Compute ResNet embedding here (maintains same training behavior)
        resnet_embed = self.resnet_branch(target_maps)  # Shape: (batch, 8)
        
        # Compute target and predicted fields
        target_field = x1-x0
        predicted_field = self(x_t, condition_param, resnet_embed)
        
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

        
    def sample(self, cdm_mass, astro_params, cdm_mass_condition, resnet_input=None, resnet_embed=None, num_steps=100, method='euler'):
        """Generate star maps from total mass maps using the learned flow"""
        self.eval()
        device = next(self.parameters()).device
        batch_size = cdm_mass_condition.size(0)
        
        # Initialize x at time t = 0
        x = cdm_mass.expand(-1,4,-1,-1).clone()
        dt = 1.0 / num_steps

        if resnet_embed is None:
            if resnet_input is not None:
                resnet_embed = self.resnet_branch(resnet_input)
            else: 
                print("Missing ResNet inputs")
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((batch_size,), i * dt, device=device)  # t ∈ [0, 1)
                combined_condition = torch.cat([t.unsqueeze(1),astro_params], dim=1)
                
                field_change = self(x, combined_condition, cdm_mass_condition, resnet_embed)
    
                if method == 'euler':
                    x = x + dt * field_change
                else:
                    raise ValueError("Only 'euler' method implemented")
        return x

