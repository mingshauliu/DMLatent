import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from data_module import EnhancedContrastiveDataModule
from enhanced_contrastive_module import CosmologyEnhancedContrastiveModel
from callbacks import HierarchicalUMAPCallback

def create_cosmology_focused_config():
    """Configuration specifically tuned for improving cosmology separation"""
    config = {
        # Base parameters
        'img_size': 256,
        'batch_size': 64,  # Could try smaller batches for more cosmology pairs
        'k_samples': 15000,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'num_workers': 4,
        
        # Model parameters
        'model_type': 'large',
        'dropout': 0.1,
        
        # Training parameters - longer training for cosmology focus
        'lr': 2e-4,  # Slightly lower LR for more stable cosmology learning
        'weight_decay': 1e-4,
        'epochs': 120,  # Longer training
        'patience': 25,  # More patience for cosmology learning
        
        # Cosmology-focused loss parameters
        'loss_type': 'cosmology_focused',
        'cosmology_temperature': 0.03,  # Lower temperature = harder separation
        'pretrained_guidance_weight': 2.0,  # Strong guidance from pretrained CNN
        'hard_negative_weight': 3.0,       # Focus heavily on hard cosmology negatives
        
        # Adaptive weight schedule for cosmology focus
        'initial_weights': [1.0, 0.5, 0.3],    # [matter, cosmology, component]
        'target_weights': [0.6, 5.0, 0.1],     # Boost cosmology to 5x, reduce others
        'cosmology_warmup_epochs': 10,         # Shorter warmup
        'cosmology_focus_epochs': 60,          # Long cosmology focus period
        
        # Data parameters
        'pair_type': 'MultiComponent',
        'blur_kernel': 0,
        'normalize': False,
        
        # Paths
        'pretrained_path': "/n/netscratch/iaifi_lab/Lab/msliu/small_net/best_cnn_model_blur_0_large.pt",
        'data_root': '/n/netscratch/iaifi_lab/Lab/',
    }
    return config

def setup_cosmology_data_paths():
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

def setup_cosmology_logging(config):
    """Setup logging specifically for cosmology-focused training"""
    exp_name = f"cosmology_focused_{config['model_type']}_guidance{config['pretrained_guidance_weight']}"
    
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="cosmology_logs",
        name=exp_name,
        version=None,
        log_graph=True,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # CSV logger  
    csv_logger = CSVLogger(
        save_dir="cosmology_logs",
        name=exp_name + "_csv",
        version=tb_logger.version,
    )
    loggers.append(csv_logger)
    
    print(f"Cosmology-focused logging to: {tb_logger.log_dir}")
    return loggers

class CosmologyProgressCallback(pl.Callback):
    """Callback to track cosmology separation progress"""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # Log current adaptive weights
        current_weights = pl_module.weight_scheduler.get_weights(
            epoch, pl_module.last_cosmology_silhouette
        )
        
        print(f"\nEpoch {epoch} - Adaptive Weights:")
        print(f"  Matter: {current_weights[0]:.2f}")
        print(f"  Cosmology: {current_weights[1]:.2f} ← FOCUS")
        print(f"  Component: {current_weights[2]:.2f}")
        print(f"  Last Cosmology Silhouette: {pl_module.last_cosmology_silhouette:.3f}")
        
        # Provide guidance based on progress
        if epoch == 10:
            print("\n🚀 PHASE 1 COMPLETE: Basic hierarchy learned")
            print("   Now boosting cosmology weight significantly...")
            
        elif epoch == 70:
            print("\n🎯 PHASE 2 COMPLETE: Cosmology focus period")
            print("   Now balancing all hierarchy levels...")
            
        # Alert if cosmology separation is improving
        if hasattr(pl_module, '_prev_cosmology_silhouette'):
            improvement = pl_module.last_cosmology_silhouette - pl_module._prev_cosmology_silhouette
            if improvement > 0.05:
                print(f"   ✅ Cosmology separation improving! (+{improvement:.3f})")
            elif improvement < -0.05:
                print(f"   ⚠️ Cosmology separation declining (-{abs(improvement):.3f})")
        
        pl_module._prev_cosmology_silhouette = pl_module.last_cosmology_silhouette

def main():
    """Main training function for cosmology-focused learning"""
    config = create_cosmology_focused_config()
    
    print("=" * 60)
    print("🔬 COSMOLOGY-FOCUSED CONTRASTIVE TRAINING")
    print("=" * 60)
    print("GOAL: Improve CDM/WDM separation using pretrained CNN guidance")
    print(f"Strategy: 3-phase training with adaptive weights")
    print(f"  Phase 1 (0-{config['cosmology_warmup_epochs']}): Normal hierarchy")
    print(f"  Phase 2 ({config['cosmology_warmup_epochs']}-{config['cosmology_warmup_epochs'] + config['cosmology_focus_epochs']}): COSMOLOGY FOCUS") 
    print(f"  Phase 3 ({config['cosmology_warmup_epochs'] + config['cosmology_focus_epochs']}+): Balanced refinement")
    print("=" * 60)
    
    # Create data module
    dm = EnhancedContrastiveDataModule(config['data_root'], config)
    dm.component_files = setup_cosmology_data_paths()
    
    # Create cosmology-enhanced model
    model = CosmologyEnhancedContrastiveModel(config)
    
    # Setup logging
    loggers = setup_cosmology_logging(config)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val/total_loss',
        patience=config['patience'],
        mode='min',
        verbose=True,
        min_delta=0.001,  # Smaller delta for cosmology learning
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val/cosmology_silhouette',  # Monitor cosmology separation directly
        save_top_k=1,
        mode='max',  # Higher silhouette = better separation
        filename='best-cosmology-separation-{epoch:02d}-{val/cosmology_silhouette:.3f}',
        save_last=True
    )
    
    # Enhanced UMAP callback with cosmology focus
    umap_callback = HierarchicalUMAPCallback(every_n_epochs=3)  # More frequent monitoring
    
    # Progress tracking callback
    progress_callback = CosmologyProgressCallback()
    
    # Trainer with cosmology-focused settings
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        callbacks=[early_stop, checkpoint, umap_callback, progress_callback],
        accelerator='auto',
        log_every_n_steps=5,  # More frequent logging
        gradient_clip_val=0.5,  # Tighter gradient clipping for stability
        precision=16,
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        val_check_interval=1.0,
        check_val_every_n_epoch=1
        # Track cosmology metrics
        # track_grad_norm=2,
    )
    
    # Log hyperparameters
    for logger in loggers:
        logger.log_hyperparams(config)
    
    print("\n🎯 KEY METRICS TO WATCH:")
    print("  train/cosmology_specific_loss  - Should decrease steadily")
    print("  train/pretrained_guidance      - Should stay > 0.5")
    print("  val/cosmology_silhouette       - Should increase (target > 0.3)")
    print("  train/weight_cosmology         - Will boost to 5x around epoch 10")
    print("\n📊 Monitoring:")
    print(f"  TensorBoard: tensorboard --logdir cosmology_logs")
    print(f"  UMAP plots: hierarchical_umap_epoch_X.png (every 3 epochs)")
    
    # Start training
    trainer.fit(model, datamodule=dm)
    
    print("\n" + "=" * 60)
    print("🎉 COSMOLOGY-FOCUSED TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model: {checkpoint.best_model_path}")
    print(f"Final cosmology silhouette: {model.last_cosmology_silhouette:.3f}")
    
    if model.last_cosmology_silhouette > 0.3:
        print("✅ SUCCESS: Good cosmology separation achieved!")
    elif model.last_cosmology_silhouette > 0.2:
        print("⚠️ PARTIAL: Some cosmology separation, may need more training")
    else:
        print("❌ NEEDS WORK: Cosmology separation still poor, try:")
        print("   - Increase pretrained_guidance_weight")
        print("   - Lower cosmology_temperature")
        print("   - Increase cosmology_focus_epochs")

if __name__ == "__main__":
    main()


# Additional utility functions for cosmology analysis

def analyze_cosmology_separation(model_path, data_module):
    """
    Analyze how well the trained model separates cosmologies
    """
    import torch
    import numpy as np
    from sklearn.metrics import classification_report
    
    # Load model
    model = CosmologyEnhancedContrastiveModel.load_from_checkpoint(model_path)
    model.eval()
    
    # Get test data
    test_loader = data_module.test_dataloader()
    
    all_embeddings = []
    all_cosmology_embeddings = []
    all_cosmologies = []
    all_matter_types = []
    
    with torch.no_grad():
        for batch in test_loader:
            x1, x2 = batch['map1'], batch['map2']
            cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2']
            matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
            
            out1 = model(x1)
            out2 = model(x2)
            
            all_embeddings.extend([out1['base'], out2['base']])
            all_cosmology_embeddings.extend([out1['cosmology'], out2['cosmology']])
            all_cosmologies.extend([cosmology1, cosmology2])
            all_matter_types.extend([matter_type1, matter_type2])
    
    # Concatenate
    embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    cosmo_embeddings = torch.cat(all_cosmology_embeddings, dim=0).cpu().numpy()
    cosmologies = torch.cat(all_cosmologies, dim=0).cpu().numpy()
    matter_types = torch.cat(all_matter_types, dim=0).cpu().numpy()
    
    # Analyze separation by matter type
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    print("=== COSMOLOGY SEPARATION ANALYSIS ===")
    
    for matter_type in [0, 1, 2]:  # Baryonic, DM, Total
        mask = matter_types == matter_type
        if not mask.any():
            continue
            
        matter_name = ['Baryonic', 'Dark Matter', 'Total Mass'][matter_type]
        
        # Silhouette scores
        base_sil = silhouette_score(embeddings[mask], cosmologies[mask])
        cosmo_sil = silhouette_score(cosmo_embeddings[mask], cosmologies[mask])
        
        print(f"\n{matter_name} Matter:")
        print(f"  Base embedding cosmology silhouette: {base_sil:.3f}")
        print(f"  Cosmology embedding silhouette: {cosmo_sil:.3f}")
        
        # K-means clustering accuracy
        kmeans = KMeans(n_clusters=2, random_state=42)
        pred_base = kmeans.fit_predict(embeddings[mask])
        pred_cosmo = kmeans.fit_predict(cosmo_embeddings[mask])
        
        # Align clusters with ground truth
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import accuracy_score
        
        def align_clusters(predictions, true_labels):
            # Create confusion matrix
            confusion = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    confusion[i, j] = np.sum((predictions == i) & (true_labels == j))
            
            # Find optimal assignment
            row_ind, col_ind = linear_sum_assignment(-confusion)
            aligned_pred = np.zeros_like(predictions)
            for i, j in zip(row_ind, col_ind):
                aligned_pred[predictions == i] = j
            return aligned_pred
        
        aligned_base = align_clusters(pred_base, cosmologies[mask])
        aligned_cosmo = align_clusters(pred_cosmo, cosmologies[mask])
        
        acc_base = accuracy_score(cosmologies[mask], aligned_base)
        acc_cosmo = accuracy_score(cosmologies[mask], aligned_cosmo)
        
        print(f"  Base embedding clustering accuracy: {acc_base:.3f}")
        print(f"  Cosmology embedding clustering accuracy: {acc_cosmo:.3f}")

def suggest_improvements(cosmology_silhouette, pretrained_alignment):
    """
    Suggest improvements based on current performance
    """
    print("\n=== IMPROVEMENT SUGGESTIONS ===")
    
    if cosmology_silhouette < 0.15:
        print("🔴 CRITICAL: Very poor cosmology separation")
        print("Try:")
        print("  - Increase pretrained_guidance_weight to 3.0+")
        print("  - Lower cosmology_temperature to 0.01") 
        print("  - Increase target cosmology weight to 8.0+")
        print("  - Check if pretrained CNN actually works on your data")
        
    elif cosmology_silhouette < 0.25:
        print("🟡 NEEDS IMPROVEMENT: Weak cosmology separation")
        print("Try:")
        print("  - Increase cosmology_focus_epochs to 80+")
        print("  - Add more hard negative mining")
        print("  - Try different cosmology_temperature values")
        
    elif cosmology_silhouette < 0.35:
        print("🟢 GOOD: Decent cosmology separation") 
        print("Fine-tuning suggestions:")
        print("  - Longer training with current settings")
        print("  - Slight increase in pretrained_guidance_weight")
        
    else:
        print("🎉 EXCELLENT: Strong cosmology separation!")
        print("  - Current approach is working well")
        print("  - Can try reducing other weights to focus more on cosmology")
    
    if pretrained_alignment < 0.3:
        print("\n⚠️ WARNING: Poor alignment with pretrained CNN")
        print("  - Check that pretrained model works on your component data")
        print("  - Verify pretrained_path is correct")
        print("  - Consider retraining pretrained model on mixed component data")