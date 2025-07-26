import torch
import random
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F

import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from utils import load_dataset
from transform import Augmentation 
from LClassifier import LClassifier 


def main():
    # === Config ===
    config = {
        # Paths to individual samples
        'sample_size': 15000,  # Number of boxes to sample
        'cdm_file':'/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy',
        'wdm_file':'/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mstar_IllustrisTNG_WDM_z=0.00.npy',
        
        # Model type
        'model_type': 'vit',  # Options: 'vit'
        # 'resume_from':'/n/netscratch/iaifi_lab/Lab/msliu/cnn_galaxy/models/medium_model.ckpt',
        # 'resume_from':'/n/netscratch/iaifi_lab/Lab/msliu/cnn_galaxy/models/tiny_model.ckpt',
        'resume_from': None,  
        
        # Training parameters
        'lr': 3e-4,
        'weight_decay': 2e-3,
        'batch_size': 32,
        'val_split': 0.2,  # 20% of test set as validation
        'max_epochs': 80,
        'patience': 20,
        'dropout': 0.3,  # Dropout rate for the model

        # Data processing
        'blur_sigma':0.3,
        'blur_kernel': 0,
        'scaling_scheme': 'log',
        'normalize': False,  # z-score normalization
    }

    # === Transform ===
    normalize = Augmentation(
        size=(256, 256),
        scaling_scheme=config.get('scaling_scheme', 'log'),
        normalize=config.get('normalize', False),
        blur_kernel=config['blur_kernel']  
    )

    # === Dataset & Split ===

    full_indices = random.sample(range(15000), k=config['sample_size'])  
    random.shuffle(full_indices)

    full_dataset = load_dataset(
        indices=full_indices, 
        transform=normalize,
        cdm_file=config['cdm_file'],
        wdm_file=config['wdm_file']
    )

    val_frac = config['val_split']
    test_frac = val_frac  # You can adjust this

    num_total = len(full_dataset)
    num_val = int(val_frac * num_total)
    num_test = int(test_frac * num_total)
    num_train = num_total - num_val - num_test

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [num_train, num_val, num_test]
    )
    
    print(f"Total dataset size: {num_total}")
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True, shuffle=False)


    # === Model ===
    if config['resume_from'] is not None:
        print(f"Resuming from checkpoint: {config['resume_from']}")
        model = LClassifier.load_from_checkpoint(
            config.get('resume_from', None),  # Resume from checkpoint if provided
            model_type=config['model_type'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            dropout=config.get('dropout',0)  # You can adjust dropout if needed
        )
    else:
        print("Training from scratch")
        model = LClassifier(
            model_type=config['model_type'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            dropout=config.get('dropout', 0)  # Default to 0 if not specified
        )

    # === Logger ===
    logger = CSVLogger("logs", name=f"{config['model_type']}_eval")

    # === Early Stopping Callback ===
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=config['patience'],
        verbose=True,
        mode="max"
    )

    # === Trainer ===
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config['max_epochs'],
        logger=logger,
        log_every_n_steps=10,
        callbacks=[early_stop_callback],
        deterministic=True,
    )

    # === Training ===
    trainer.fit(model, train_loader, val_loader)
    
    # === Save Model ===
    model_path = f"models/{config['model_type']}_model.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")
    print(f"Training complete. Best model saved to {model_path}")
    
    # === Validation and Test ===
    trainer.validate(model, dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    
    # === Feature Extraction with Soft Scores ===
    model.eval()
    model.to("cuda")

    features = []
    softscores = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to("cuda")

            # Get features before final FC
            feats = model.model.forward_features(x)  # [B, C, H, W]
            features.append(feats.cpu())

            # Soft scores from final output
            logits = model.model.fc(feats).squeeze(1)  # [B]
            probs = torch.sigmoid(logits)
            softscores.append(probs.cpu())

            labels.append(y.cpu())

    # === Stack Results ===
    features = torch.cat(features).numpy()
    softscores = torch.cat(softscores).numpy()
    labels = torch.cat(labels).numpy()

    # === UMAP ===
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    scaled = StandardScaler().fit_transform(features)
    embedding = reducer.fit_transform(scaled)

    # === Plot: UMAP colored by soft score
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=softscores, cmap='viridis', s=10, alpha=0.7)
    plt.title("Galactic scale WDM/CDM latent space (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(label="Soft Score (Sigmoid Output)")
    plt.tight_layout()
    plt.savefig(f"umap_softscore_{config['model_type']}.png")
    plt.close()

    
    # === Feature Map Visualization ===
    sample_x, sample_y = next(iter(test_loader))
    sample_x = sample_x.to("cuda")[:4]  # Take 4 samples
    with torch.no_grad():
        fmap = model.model.features[0](sample_x)  # First conv layer output
        fmap = fmap.cpu()

    # Plot first 4 feature maps for the first image
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axes[i].imshow(fmap[0, i].numpy(), cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"Feature {i}")
    plt.suptitle("First Layer Feature Maps (Image 0)")
    plt.tight_layout()
    plt.savefig(f"feature_maps_{config['model_type']}.png")
    plt.close()

if __name__ == "__main__":
    pl.seed_everything(42)  # For reproducibility
    main()