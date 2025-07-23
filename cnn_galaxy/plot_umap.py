import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from utils import load_multiple_hdf5_datasets
from transform import SimpleResize
from LClassifier import LClassifier

def main():
    config = {
        'sample_size': 100,
        'model_type': 'tiny_downsample',
        'resume_from': '/n/netscratch/iaifi_lab/Lab/msliu/cnn_galaxy/models/tiny_downsample_model.ckpt',
        'batch_size': 64,
        'val_split': 0.2,
        'dropout': 0.2,
        'scaling_scheme': 'tanh',
    }

    # === Transform ===
    transform = SimpleResize(size=(256, 256), scaling_scheme=config['scaling_scheme'], normalize=False)

    # === Dataset ===
    full_dataset = load_multiple_hdf5_datasets(
        indices=random.sample(range(1024), k=config['sample_size']),
        transform=transform,
        base_path='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images'
    )

    # === DataLoader ===
    full_loader = DataLoader(full_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=False)

    # === Load Model ===
    model = LClassifier.load_from_checkpoint(
        config['resume_from'],
        model_type=config['model_type'],
        lr=3e-4,  # Dummy, not used during eval
        weight_decay=1e-3,
        dropout=config['dropout']
    )
    model.eval()
    model.to("cuda")

    # === Feature Extraction with Soft Scores ===
    features, softscores, labels = [], [], []
    with torch.no_grad():
        for x, y in full_loader:
            x = x.to("cuda")
            feats = model.model.features(x)
            pooled = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
            logits = model.model.classifier(feats).squeeze(1)
            probs = torch.sigmoid(logits)

            features.append(pooled.cpu())
            softscores.append(probs.cpu())
            labels.append(y.cpu())

    features = torch.cat(features).numpy()
    softscores = torch.cat(softscores).numpy()
    labels = torch.cat(labels).numpy()
    
    from sklearn.metrics import accuracy_score

    # Convert soft scores to hard predictions
    preds = (softscores > 0.5).astype(int)
    acc = accuracy_score(labels, preds)

    # === UMAP ===
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    scaled = StandardScaler().fit_transform(features)
    embedding = reducer.fit_transform(scaled)

    # === Plot: UMAP colored by soft score
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=softscores, cmap='viridis', s=5, alpha=0.7)
    plt.title(f"Galactic-scale WDM/CDM latent space ({acc*100:.2f}%)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(label="Soft Score (Sigmoid Output)")
    plt.tight_layout()
    plt.savefig("umap_softscore_big.png")
    plt.close()


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
