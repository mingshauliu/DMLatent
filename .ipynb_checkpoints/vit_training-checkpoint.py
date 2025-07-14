"""
Vision Transformer for CDM/WDM Classification
Clean version for cluster submission
"""

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms.functional import gaussian_blur
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import math


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class HDF5KeyImageDataset(Dataset):
    """Dataset class for loading CDM/WDM HDF5 files"""
    
    def __init__(self, cdm_path, wdm_path, transform=None):
        self.transform = transform
        self.data_sources = []  # List of (file_type, key_name, sub_index_if_needed)
        
        # Load CDM keys
        self.cdm_file = h5py.File(cdm_path, 'r')
        for key in self.cdm_file.keys():
            self.data_sources.append(('cdm', key, None))

        # Load WDM keys
        self.wdm_file = h5py.File(wdm_path, 'r')
        for key in self.wdm_file.keys():
            self.data_sources.append(('wdm', key, None))

    def __len__(self):
        return len(self.data_sources)

    def __getitem__(self, idx):
        source, key, sub_idx = self.data_sources[idx]
        h5_file = self.cdm_file if source == 'cdm' else self.wdm_file
        label = 0 if source == 'cdm' else 1
    
        arr = h5_file[key]
        img = arr[:] if sub_idx is None else arr[sub_idx]
    
        # Clean and normalize
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img = np.clip(img, 0, None)  # Ensure non-negative
        img = img.astype(np.float32)
    
        if img.max() > 0:
            img = img / img.max()  # Normalize to [0, 1]
    
        # Convert to torch tensor and add channel dim
        img = torch.from_numpy(img).unsqueeze(0)  # shape: [1, H, W]
    
        if self.transform:
            img = self.transform(img)
    
        return img, torch.tensor(label, dtype=torch.float32)

    def __del__(self):
        if hasattr(self, 'cdm_file'):
            self.cdm_file.close()
        if hasattr(self, 'wdm_file'):
            self.wdm_file.close()


def gaussian_blur_tensor(tensor, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to tensor"""
    if tensor.dim() == 3:  # [C, H, W]
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]
    return gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)


class TensorAugment(nn.Module):
    """Data augmentation module for tensors"""
    
    def __init__(self, size=(256, 256), p_flip=0.5, p_rot=0.5,
                 noise_std=0.01, apply_blur=False, blur_kernel=5, blur_sigma=1.0):
        super().__init__()
        self.size = size
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.noise_std = noise_std
        self.apply_blur = apply_blur
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def forward(self, img):  # img: [1, H, W] or [H, W]
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        # Resize to target size
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        img = img.squeeze(0)  # [1, H, W]

        # Random flips and rotation
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[2])  # H-flip
        if random.random() < self.p_flip:
            img = torch.flip(img, dims=[1])  # V-flip
        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            img = torch.rot90(img, k=k, dims=[1, 2])

        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise

        # Optional blur
        if self.apply_blur:
            img = gaussian_blur_tensor(img, kernel_size=self.blur_kernel, sigma=self.blur_sigma).squeeze(0)

        return img


class SimpleResize(nn.Module):
    """Simple resize transform for validation/test sets (no augmentation)"""
    
    def __init__(self, size=(256, 256)):
        super().__init__()
        self.size = size

    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        return img.squeeze(0)


def load_multiple_hdf5_datasets(indices, transform=None, base_path='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images'):
    """Load multiple HDF5 datasets and concatenate them"""
    datasets = []
    for idx in indices:
        cdm_path = f'{base_path}/CDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        wdm_path = f'{base_path}/WDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        
        if os.path.exists(cdm_path) and os.path.exists(wdm_path):
            ds = HDF5KeyImageDataset(cdm_path, wdm_path, transform=transform)
            datasets.append(ds)
        else:
            print(f"Warning: Skipping missing files for box_{idx}")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    return ConcatDataset(datasets)


class PatchEmbed(nn.Module):
    """Patch embedding layer"""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                     # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)     # [B, N, embed_dim]
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        self.attn_weights = attn_weights.detach()
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SimpleViT(nn.Module):
    """Simple Vision Transformer for binary classification"""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=1, num_classes=1,
                 embed_dim=128, depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size)**2 + 1, embed_dim))
        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, dropout) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        for block in self.transformer:
            x = block(x)
        return self.head(x[:, 0])

    def get_last_attention_map(self):
        """Get attention weights from the last transformer block"""
        return self.transformer[-1].attn_weights


def plot_training_progress(train_loss, train_acc, val_acc, save_path='training_progress.png'):
    """Plot training progress"""
    epochs = range(1, len(train_loss) + 1)
    best_epoch = val_acc.index(max(val_acc)) + 1

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Training loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color='tab:red')
    ax1.plot(epochs, train_loss, color='tab:red', label="Train Loss", linestyle='--', marker='x')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:blue')
    ax2.plot(epochs, train_acc, label="Train Accuracy", color='tab:orange', linestyle='--')
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color='tab:green', linestyle='--', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Highlight best validation epoch
    ax2.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax2.plot(best_epoch, val_acc[best_epoch - 1], 'o', color='tab:green', markersize=8,
             label=f"Best Val Epoch ({best_epoch})")

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4)

    plt.title("Training Loss and Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_map(attn_map, patch_size=16, img_size=256, head=0, save_path='attention_map.png'):
    """Visualize attention from CLS token to image patches"""
    H, N, _ = attn_map.shape
    cls_attn = attn_map[head, 0, 1:]  # CLS token's attention to all patches

    num_patches = int(math.sqrt(N - 1))
    cls_attn_grid = cls_attn.reshape(num_patches, num_patches).cpu()

    # Upsample to image size
    full_map = F.interpolate(cls_attn_grid.unsqueeze(0).unsqueeze(0),
                             size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze()

    plt.figure(figsize=(8, 6))
    plt.imshow(full_map, cmap='viridis')
    plt.title(f"Attention Map (Head {head})")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on given data loader"""
    model.eval()
    total_loss, correct = 0.0, 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return avg_loss, accuracy


def main():
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        'img_size': 256,
        'patch_size': 16,
        'embed_dim': 128,
        'depth': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'epochs': 60,
        'patience': 10,
        'k_samples': 50,  # Number of samples to use
    }
    
    print("=== Vision Transformer Training ===")
    print(f"Configuration: {config}")
    
    # Load WDM mass data (if needed)
    try:
        WDM_mass = []
        with open('WDM_TNG_MW_SB4.txt', 'r') as f:
            for i, line in enumerate(f.readlines()[1:]):
                WDM_mass.append([i, float(line.strip().split(' ')[0])])
        WDM_mass = np.array(WDM_mass)
        print(f"Loaded WDM mass data: {len(WDM_mass)} entries")
    except FileNotFoundError:
        print("Warning: WDM_TNG_MW_SB4.txt not found, using random indices")
        WDM_mass = None
    
    # Sample indices
    all_indices = random.sample(range(1024), config['k_samples'])
    random.shuffle(all_indices)
    
    # Split data
    train_ratio, val_ratio = 0.6, 0.2
    total_samples = len(all_indices)
    train_end = int(train_ratio * total_samples)
    val_end = int((train_ratio + val_ratio) * total_samples)
    
    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:]
    
    print(f"\nDataset split:")
    print(f"Total: {total_samples}, Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create transforms
    train_transform = TensorAugment(
        size=(config['img_size'], config['img_size']),
        p_flip=0.5,
        p_rot=0.5,
        noise_std=0.005,
        apply_blur=False
    )
    
    val_test_transform = SimpleResize(size=(config['img_size'], config['img_size']))
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = load_multiple_hdf5_datasets(train_indices, transform=train_transform)
    val_dataset = load_multiple_hdf5_datasets(val_indices, transform=val_test_transform)
    test_dataset = load_multiple_hdf5_datasets(test_indices, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleViT(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_chans=1,
        num_classes=1,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training tracking
    train_loss_hist, train_acc_hist, val_acc_hist = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    
    # Training loop
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation phase
        avg_val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(avg_val_loss)
        
        # Record history
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        
        print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'config': config
            }, "best_vit_model.pt")
            print(f"  -> New best model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load("best_vit_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\nFinal Results:")
    print(f"Best Val Acc: {max(val_acc_hist):.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    # Plot training progress
    plot_training_progress(train_loss_hist, train_acc_hist, val_acc_hist, 'training_progress.png')
    
    # Generate attention map example
    if len(test_loader) > 0:
        model.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                images = images[:1].to(device)  # Take first image
                _ = model(images)
                attn_map = model.get_last_attention_map()
                if attn_map is not None:
                    plot_attention_map(attn_map, save_path='attention_example.png')
                    print("Attention map saved as 'attention_example.png'")
                break
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()