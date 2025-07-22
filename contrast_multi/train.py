import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from data_module import EnhancedContrastiveDataModule
from contrastive_module import HierarchicalContrastiveModel
from callbacks import HierarchicalUMAPCallback
import os

def setup_logging(config):
    """Setup comprehensive logging for hierarchical training"""
    
    # Create experiment name
    exp_name = f"hierarchical_{config['loss_type']}_{config['model_type']}_bs{config['batch_size']}"
    
    # Setup multiple loggers
    loggers = []
    
    # TensorBoard logger for rich visualizations
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=exp_name,
        version=None,  # Auto-increment
        log_graph=True,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # CSV logger for easy data analysis
    csv_logger = CSVLogger(
        save_dir="lightning_logs",
        name=exp_name + "_csv",
        version=tb_logger.version,  # Match version numbers
    )
    loggers.append(csv_logger)
    
    print(f"Logging to: {tb_logger.log_dir}")
    print(f"TensorBoard command: tensorboard --logdir {os.path.abspath('lightning_logs')}")
    
    return loggers

def create_hierarchical_config():
    """Configuration for hierarchical contrastive learning"""
    config = {
        # Data parameters
        'img_size': 256,
        'batch_size': 64,
        'k_samples': 15000,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'num_workers': 4,
        
        # Model parameters  
        'model_type': 'large',
        'dropout': 0.1,
        
        # Training parameters
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'epochs': 80,
        'patience': 15,  # Longer patience for hierarchical learning
        
        # Hierarchical contrastive learning parameters
        'pair_type': 'MultiComponent',  # Use multi-component pairs
        'loss_type': 'multilevel',  # Options: 'hierarchical_nxtent', 'adaptive_hierarchical', 'multilevel'
        'temperature': 0.05,
        
        # Hierarchy weights - adjust based on your priorities
        'matter_weight': 2.0,      # Strongest: separate baryonic vs DM vs total
        'cosmology_weight': 1.0,   # Medium: separate CDM vs WDM within matter types
        'component_weight': 0.5,   # Weakest: separate gas vs stars within baryonic
        
        # For multilevel loss
        'level_weights': [2.0, 1.0, 0.5],  # [matter, cosmology, component]
        
        # Auxiliary supervision (optional)
        'use_auxiliary_classifier': False,
        'alignment_weight': 0.1,
        
        # Augmentation parameters
        'blur_kernel': 0,
        'normalize': False,
        
        # Paths - UPDATE THESE TO YOUR ACTUAL PATHS
        'pretrained_path': "/n/netscratch/iaifi_lab/Lab/msliu/small_net/best_cnn_model_blur_0_large.pt",
        'data_root': '/n/netscratch/iaifi_lab/Lab/',
    }
    return config

def setup_hierarchical_data_paths():
    return {
        'gas_cdm': '/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy',
        'gas_wdm': '/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mgas_IllustrisTNG_WDM_z=0.00.npy',
        'stars_cdm': '/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy',
        'stars_wdm': '/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mstar_IllustrisTNG_WDM_z=0.00.npy',
        'dm_cdm': '/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy',
        'dm_wdm': '/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mcdm_IllustrisTNG_WDM_z=0.00.npy',
        'total_cdm': '/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy',
        'total_wdm': '/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mtot_IllustrisTNG_WDM_z=0.00.npy'
    }

def main():
    """Main training function"""
    config = create_hierarchical_config()
    
    # Create data module
    dm = EnhancedContrastiveDataModule(config['data_root'], config)
    
    # Override component files with your actual paths
    dm.component_files = setup_hierarchical_data_paths()
    
    # Create hierarchical model
    model = HierarchicalContrastiveModel(config)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=config['patience'], 
        mode='min',
        verbose=True
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss', 
        save_top_k=1, 
        mode='min', 
        filename='best-hierarchical-contrastive',
        save_last=True
    )
    
    # Hierarchical UMAP visualization callback
    umap_callback = HierarchicalUMAPCallback(every_n_epochs=5)
    
    # Setup logging
    loggers = setup_logging(config)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        callbacks=[early_stop, checkpoint, umap_callback],
        accelerator='auto',
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Helpful for hierarchical losses
        precision=16,  # Mixed precision for efficiency
        logger=loggers,  # Use comprehensive logging
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        # Additional useful options
        val_check_interval=1.0,  # Validate every epoch
        check_val_every_n_epoch=1,
        sync_batchnorm=True,  # For multi-GPU
    )
    
    print("=== Starting Hierarchical Contrastive Training ===")
    print(f"Loss type: {config['loss_type']}")
    print(f"Pair type: {config['pair_type']}")
    print(f"Hierarchy weights: matter={config['matter_weight']}, cosmology={config['cosmology_weight']}, component={config['component_weight']}")
    print(f"Logging directory: {loggers[0].log_dir}")
    
    # Log hyperparameters to all loggers
    for logger in loggers:
        logger.log_hyperparams(config)
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    print("=== Training Complete ===")
    print(f"Best model saved to: {checkpoint.best_model_path}")

if __name__ == "__main__":
    main()
