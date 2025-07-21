import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# ==============================
# FiLM Layer (Feature-wise Linear Modulation)
# ==============================
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
            nn.ReLU()
        )

    def forward(self, features, cond):
        gamma_beta = self.condition_proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


# ==============================
# Flow Matching Vector Field
# ==============================
class ConditionalFlowNet(nn.Module):
    def __init__(self, in_channels=1, cond_dim=2, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.film = FiLMLayer(hidden_dim, cond_dim)
        self.out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x_t, t, cond):
        B, _, H, W = x_t.shape
        t = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        inp = torch.cat([x_t, t], dim=1)
        h = self.encoder(inp)
        h = self.film(h, cond)
        return self.out(h)


# ==============================
# Dataset
# ==============================
class MassBaryonDataset(Dataset):
    def __init__(self, mass_maps, baryon_maps, labels):
        self.mass_maps = mass_maps
        self.baryon_maps = baryon_maps
        self.labels = labels

    def __len__(self):
        return len(self.mass_maps)

    def __getitem__(self, idx):
        x = torch.tensor(self.mass_maps[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.baryon_maps[idx], dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        cond = F.one_hot(torch.tensor(label), num_classes=2).float()
        return {"mass_map": x, "baryon_map": y, "condition_vector": cond}


# ==============================
# Lightning Module
# ==============================
class FlowMatchingModule(pl.LightningModule):
    def __init__(self, cond_dim=2, lr=1e-3):
        super().__init__()
        self.model = ConditionalFlowNet(in_channels=1, cond_dim=cond_dim)
        self.lr = lr

    def interpolate(self, x, y, t):
        return (1 - t) * x + t * y

    def flow_matching_loss(self, v_pred, x, y):
        target = y - x
        return F.mse_loss(v_pred, target)

    def forward(self, x_t, t, cond):
        return self.model(x_t, t, cond)

    def training_step(self, batch, batch_idx):
        x = batch['mass_map']
        y = batch['baryon_map']
        cond = batch['condition_vector']
        t = torch.rand(x.size(0), device=self.device)
        x_t = self.interpolate(x, y, t.view(-1, 1, 1, 1))
        v_pred = self(x_t, t, cond)
        loss = self.flow_matching_loss(v_pred, x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['mass_map']
        y = batch['baryon_map']
        cond = batch['condition_vector']
        t = torch.rand(x.size(0), device=self.device)
        x_t = self.interpolate(x, y, t.view(-1, 1, 1, 1))
        v_pred = self(x_t, t, cond)
        loss = self.flow_matching_loss(v_pred, x, y)
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            with torch.no_grad():
                grid = vutils.make_grid(v_pred, normalize=True, scale_each=True)
                self.logger.experiment.add_image("val/flow_prediction", grid, self.global_step)
                grid_xt = vutils.make_grid(x_t, normalize=True, scale_each=True)
                self.logger.experiment.add_image("val/input_xt", grid_xt, self.global_step)
                grid_y = vutils.make_grid(y - x, normalize=True, scale_each=True)
                self.logger.experiment.add_image("val/target_delta", grid_y, self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ==============================
# DataModule
# ==============================
class MassBaryonDataModule(pl.LightningDataModule):
    def __init__(self, mass_maps, baryon_maps, labels, batch_size=32, split=0.8):
        super().__init__()
        self.mass_maps = mass_maps
        self.baryon_maps = baryon_maps
        self.labels = labels
        self.batch_size = batch_size
        self.split = split

    def setup(self, stage=None):
        N = len(self.mass_maps)
        idx = int(self.split * N)
        self.train_dataset = MassBaryonDataset(self.mass_maps[:idx], self.baryon_maps[:idx], self.labels[:idx])
        self.val_dataset = MassBaryonDataset(self.mass_maps[idx:], self.baryon_maps[idx:], self.labels[idx:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
