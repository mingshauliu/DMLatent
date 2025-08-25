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

def train_flow_matching_model(cdm_mass_maps, star_maps, gas_maps, T_maps, astro_params,
                            architecture='unet',
                            noise_std=0.0,
                            max_epochs=100,
                            batch_size=16,
                            patience=20):
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    data_module = AstroFlowMatchingDataModule(
        cdm_mass_maps=cdm_mass_maps,
        star_maps=star_maps,
        gas_maps=gas_maps,
        T_maps=T_maps,
        astro_params=astro_params,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=4  # Back to 4 workers for faster data loading
    )
    
    model = FlowMatchingModel(
        architecture=architecture,
        noise_std=noise_std,
        learning_rate=1e-4,
        alpha=10
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
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

    # ckpt_path = None
    # ckpt_dir = '/n/netscratch/iaifi_lab/Lab/msliu/flow_COND/lightning_logs/tng_2param/checkpoints/'
    # if os.path.isdir(ckpt_dir):
    #     ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    #     if ckpts:
    #         ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])  # load latest checkpoint
    #         print(f"Resuming from checkpoint: {ckpt_path}")
    #     else:
    #         print("No checkpoint found. Training from scratch.")

    
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
        callbacks=[early_stop, checkpoint],
        accumulate_grad_batches=1,  # No need for gradient accumulation with larger batch size
        strategy='auto',
        enable_progress_bar=True,
        enable_model_summary=True,  # Re-enable model summary
        enable_checkpointing=True,
        detect_anomaly=False,  # Keep disabled to save some memory
        use_distributed_sampler=False,
        limit_train_batches=1.0,  # Use full training set
        limit_val_batches=1.0,    # Use full validation set
        num_sanity_val_steps=2    # Re-enable sanity check
    )
    
    # trainer.fit(model, data_module, ckpt_path=ckpt_path)
    trainer.fit(model, data_module)
    
    print(f"Best model saved at: {checkpoint.best_model_path}")
    
    return model, trainer

if __name__ == "__main__":

    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    config={
        'models': ['IllustrisTNG', 'EAGLE', 'SIMBA', 'Astrid'],  # Back to 4 models
        'samples_per_model': 8000,  # Number of samples to load from each model
        'noise_std': 0.2,
        'architecture': 'unet',
        'max_epochs': 200,
        'batch_size': 32,  # Back to larger batch size for speed
        'patience': 30
    }
    print('Configurations:',config)
    
    # Load data from multiple models
    all_cdm_mass_maps = []
    all_star_maps = []
    all_gas_maps = []
    all_T_maps = []
    all_astro_params = []
    
    for model_name in config['models']:
        print(f"Loading data from {model_name}...")
        
        cdm_mass = np.load(f'/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/{model_name}/Maps_Mcdm_{model_name}_LH_z=0.00.npy')
        astro_params = np.loadtxt(f'/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/{model_name}/params_LH_{model_name}.txt')
        star_maps = np.load(f'/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/{model_name}/Maps_Mstar_{model_name}_LH_z=0.00.npy')
        gas_maps = np.load(f'/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/{model_name}/Maps_Mgas_{model_name}_LH_z=0.00.npy')
        T_maps = np.load(f'/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/{model_name}/Maps_T_{model_name}_LH_z=0.00.npy')
        
        # Randomly sample specified number of samples from each model
        n_samples = min(config['samples_per_model'], len(cdm_mass))
        if len(cdm_mass) > n_samples:
            # Randomly sample indices
            indices = np.random.choice(len(cdm_mass), n_samples, replace=False)
            cdm_mass = cdm_mass[indices]
            star_maps = star_maps[indices]
            gas_maps = gas_maps[indices]
            T_maps = T_maps[indices]
            astro_params = astro_params[indices//15]
        
        print(f"  Loaded {len(cdm_mass)} samples from {model_name}")
        
        # Apply log1p transformation
        cdm_mass = np.log1p(cdm_mass)
        star_maps = np.log1p(star_maps)
        gas_maps = np.log1p(gas_maps)
        T_maps = np.log1p(T_maps)
        astro_params = astro_params[:,:2]
        
        # Add model identifier to astro_params (optional, for debugging)
        # model_id = np.full((astro_params.shape[0], 1), config['models'].index(model_name))
        # astro_params = np.hstack([astro_params, model_id])
        
        all_cdm_mass_maps.append(cdm_mass)
        all_star_maps.append(star_maps)
        all_gas_maps.append(gas_maps)
        all_T_maps.append(T_maps)
        all_astro_params.append(astro_params)
    
    # Combine all data
    cdm_mass_maps = np.concatenate(all_cdm_mass_maps, axis=0)
    star_maps = np.concatenate(all_star_maps, axis=0)
    gas_maps = np.concatenate(all_gas_maps, axis=0)
    T_maps = np.concatenate(all_T_maps, axis=0)
    astro_params = np.concatenate(all_astro_params, axis=0)
    
    # Alternative: Progressive loading strategy for very large datasets
    # If still running out of memory, you can:
    # 1. Load models one by one and train incrementally
    # 2. Use data streaming with torch.utils.data.IterableDataset
    # 3. Implement gradient checkpointing in the model
    
    # Memory cleanup before training
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Combined dataset sizes:")
    print(f"  Total mass maps: {cdm_mass_maps.shape}")
    print(f"  Star maps: {star_maps.shape}")
    print(f"  Gas maps: {gas_maps.shape}")
    print(f"  Temperature maps: {T_maps.shape}")
    print(f"  Astro params: {astro_params.shape}")
    print(f"  Training on {len(config['models'])} models: {config['models']}")
    
    # Estimate memory usage
    total_samples = cdm_mass_maps.shape[0]
    map_size = cdm_mass_maps.shape[1] * cdm_mass_maps.shape[2]
    estimated_memory_gb = (total_samples * map_size * 4 * 3) / (1024**3)  # Rough estimate for 3 data types
    print(f"Estimated memory usage: ~{estimated_memory_gb:.2f} GB")
    print(f"Batch size: {config['batch_size']}, Effective batch size: {config['batch_size']} (no gradient accumulation)")
    print(f"Samples per model: {config['samples_per_model']}, Total samples: {total_samples}")
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory available: {gpu_memory:.2f} GB")
        if estimated_memory_gb > gpu_memory * 0.8:
            print("⚠️  WARNING: Estimated memory usage is high relative to GPU memory!")
            print("   Consider reducing samples_per_model or batch_size further.")
        else:
            print("✅ Memory usage looks good for training!")

    print("Training U-Net Flow Matching Model on multiple models...")
    model_unet, trainer_unet = train_flow_matching_model(
        cdm_mass_maps, star_maps, gas_maps, T_maps, astro_params,
        noise_std=config['noise_std'],
        architecture=config['architecture'],
        max_epochs=config['max_epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    print("Training complete!")