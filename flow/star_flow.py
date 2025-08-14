import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

import os

from module import AstroFlowMatchingDataModule, FlowMatchingModel

def train_flow_matching_model(total_mass_maps, star_maps, 
                            architecture='unet',
                            noise_std=0.5,
                            max_epochs=100,
                            batch_size=16,
                            patience=15):
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    data_module = AstroFlowMatchingDataModule(
        total_mass_maps=total_mass_maps,
        star_maps=star_maps,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=4
    )
    
    model = FlowMatchingModel(
        architecture=architecture,
        noise_std=noise_std,
        learning_rate=1e-4,
        alpha=10
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filename='best-model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    ckpt_path = None
    # ckpt_dir = '/n/netscratch/iaifi_lab/Lab/msliu/flow/lightning_logs/9mzh1db9/checkpoints/'
    # if os.path.isdir(ckpt_dir):
    #     ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    #     if ckpts:
    #         ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])  # load latest checkpoint
    #         print(f"Resuming from checkpoint: {ckpt_path}")
    #     else:
    #         print("No checkpoint found. Training from scratch.")
    # else:
    #     print("No checkpoint found. Training from scratch.")

    
    logger = WandbLogger(log_model="False")
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        callbacks=[early_stop, checkpoint]
    )
    
    # trainer.fit(model, data_module, ckpt_path=ckpt_path)
    trainer.fit(model, data_module)
    
    print(f"Best model saved at: {checkpoint.best_model_path}")
    
    return model, trainer

if __name__ == "__main__":

    total_mass_maps = np.load('/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy')
    star_maps = np.load('/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy')

    total_mass_maps = np.log1p(total_mass_maps)
    star_maps = np.log1p(star_maps)

    config = {
        'noise_std': 0.0,
        'architecture': 'unet',
        'max_epochs': 200,
        'batch_size': 32
    }
    
    print('Configuration: ',config)
    
    print("Training U-Net Flow Matching Model...")
    model_unet, trainer_unet = train_flow_matching_model(
        total_mass_maps, star_maps, 
        noise_std=config['noise_std'],
        architecture=config['architecture'],
        max_epochs=config['max_epochs'],
        batch_size=config['batch_size']
    )
    
    print("Training complete!")