"""
CNN for CDM/WDM Classification
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
import time

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
        self.cdm_file = h5py.File(cdm_path)
        for key in self.cdm_file.keys():
            self.data_sources.append(('cdm', key, None))

        # Load WDM keys
        self.wdm_file = h5py.File(wdm_path)
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

class LogTransform:
    def __call__(self, tensor):
        return torch.log1p(tensor)

class TensorAugment(nn.Module):
    """Data augmentation module for tensors"""
    
    def __init__(self, size=(256, 256), p_flip=0.5, p_rot=0.5,
                 noise_std=0.01, apply_blur=False, apply_log=True, blur_kernel=5, blur_sigma=1.0):
        super().__init__()
        self.size = size
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.noise_std = noise_std
        self.apply_log = apply_log
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

        # Optional log scale
        if self.apply_log:
            img = torch.clamp(img, min=0)
            img = torch.log1p(img)

        return img


class SimpleResize(nn.Module):
    """Simple resize transform for validation/test sets (no augmentation)"""
    
    def __init__(self, size=(256, 256), apply_log=True):
        super().__init__()
        self.size = size
        self.apply_log = apply_log

    def forward(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 1:
            img = img[:1]

        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        if self.apply_log:
            img = torch.clamp(img, min=0)
            img = torch.log1p(img)
        return img.squeeze(0)


# def load_multiple_hdf5_datasets(indices, transform=None, base_path='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images'):
#     """Load multiple HDF5 datasets and concatenate them"""
#     datasets = []
#     for idx in indices:
#         cdm_path = f'{base_path}/CDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
#         wdm_path = f'{base_path}/WDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        
#         if os.path.exists(cdm_path) and os.path.exists(wdm_path):
#             ds = HDF5KeyImageDataset(cdm_path, wdm_path, transform=transform)
#             datasets.append(ds)
#         else:
#             print(f"Warning: Skipping missing files for box_{idx}")
    
#     if not datasets:
#         raise ValueError("No valid datasets found!")
    
#     return ConcatDataset(datasets)

def load_multiple_hdf5_datasets(indices, transform=None, base_path='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images'):
    """Load multiple HDF5 datasets and concatenate them, skipping corrupted or missing files"""
    datasets = []
    for idx in indices:
        cdm_path = f'{base_path}/CDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        wdm_path = f'{base_path}/WDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        
        if not (os.path.exists(cdm_path) and os.path.exists(wdm_path)):
            print(f"Warning: Skipping missing files for box_{idx}")
            continue
        
        try:
            # Attempt to open files to detect corruption
            with h5py.File(cdm_path, 'r') as f:
                _ = list(f.keys())
            with h5py.File(wdm_path, 'r') as f:
                _ = list(f.keys())
        except Exception as e:
            print(f"Warning: Skipping corrupted file for box_{idx}: {e}")
            continue

        try:
            ds = HDF5KeyImageDataset(cdm_path, wdm_path, transform=transform)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Skipping box_{idx} due to dataset init failure: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    return ConcatDataset(datasets)


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Residual block for deeper CNNs"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        return out


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification"""
    
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            ConvBlock(in_channels, 32, dropout=dropout),
            ConvBlock(32, 32, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 256x256 -> 128x128
            
            # Second block
            ConvBlock(32, 64, dropout=dropout),
            ConvBlock(64, 64, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            
            # Third block
            ConvBlock(64, 128, dropout=dropout),
            ConvBlock(128, 128, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Fourth block
            ConvBlock(128, 256, dropout=dropout),
            ConvBlock(256, 256, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
        )
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 16x16 -> 1x1
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BigCNN(nn.Module):
    """Bigger CNN with residual connections for deeper networks"""
    
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3, dropout=0)
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # 256x256 -> 64x64
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)    # 64x64
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)   # 32x32
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)  # 16x16
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)  # 8x8
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dropout=0.1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.classifier(x)
        return x


def plot_training_progress(train_loss, train_acc, val_acc, save_path='training_progress_cnn.png'):
    """Plot training progress"""
    epochs = range(1, len(train_loss) + 1)
    best_epoch = val_acc.index(max(val_acc)) + 1

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Training loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color='tab:red')
    ax1.plot(epochs, train_loss, color='tab:red', label="Train Loss", linestyle='--', marker='x')
    ax1.plot(epochs, val_acc, label="Validation Accuracy", color='tab:green', linestyle='--', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:blue')
    ax2.plot(epochs, train_acc, label="Train Accuracy", color='tab:orange', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Highlight best validation epoch
    ax1.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax1.plot(best_epoch, val_acc[best_epoch - 1], 'o', color='tab:green', markersize=8,
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


def plot_feature_maps(model, data_loader, device, save_path='feature_maps_cnn.png'):
    """Visualize feature maps from the first convolutional layer"""
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images[:1].to(device)  # Take first image
            
            # Get feature maps from first conv layer
            if hasattr(model, 'features'):
                # For SimpleCNN
                x = model.features[0].conv(images)  # First conv layer
            else:
                # For BigCNN
                x = model.conv1.conv(images)
            
            # Plot original image and some feature maps
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            
            # Original image
            orig_img = images[0, 0].cpu().numpy()
            axes[0, 0].imshow(orig_img, cmap='viridis')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            # Feature maps
            for i in range(min(15, x.shape[1])):
                row = i // 8
                col = (i % 8) + (1 if row == 0 else 0)
                if col < 8:
                    fmap = x[0, i].cpu().numpy()
                    axes[row, col].imshow(fmap, cmap='viridis')
                    axes[row, col].set_title(f'Feature {i}')
                    axes[row, col].axis('off')
            
            # Hide unused subplots
            for i in range(16):
                row = i // 8
                col = i % 8
                if i >= min(15, x.shape[1]) + 1:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Feature maps saved as '{save_path}'")
            break


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
        'dropout': 0.2,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'epochs': 80,
        'patience': 15,
        'k_samples': 512,  # Number of samples to use
        'model_type': 'simple',  # 'simple' or 'big'
    }
    
    print("=== CNN Training ===")
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
    
    # Create transforms (keeping log scale and augmentation intact)
    train_transform = TensorAugment(
        size=(config['img_size'], config['img_size']),
        p_flip=0.5,
        p_rot=0.5,
        noise_std=0, # Stop noise temporarily
        apply_blur=False,
        apply_log=True  # Keep log scale
    )
    
    val_test_transform = SimpleResize(
        size=(config['img_size'], config['img_size']),
        apply_log=True  # Keep log scale
    )
    
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
    
    if config['model_type'] == 'simple':
        model = SimpleCNN(
            in_channels=1,
            num_classes=1,
            dropout=config['dropout']
        ).to(device)
        print("Using SimpleCNN")
    else:
        model = BigCNN(
            in_channels=1,
            num_classes=1,
            dropout=config['dropout']
        ).to(device)
        print("Using BigCNN")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # Training tracking
    train_loss_hist, train_acc_hist, val_loss_hist = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_epoch = 0
    if os.path.exists("best_cnn_model_1.pt"):
        print("Resuming from checkpoint...")
        checkpoint = torch.load("best_cnn_model_1.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch} with val_loss={best_val_loss:.4f}")

    
    print(f"\nStarting training for {config['epochs']} epochs...")
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        start_time = time.time()
        num_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            
            batch_time = time.time() - batch_start
            if batch_idx == 0:
                print(f"  First batch time: {batch_time:.3f}s")
        
        total_time = time.time() - start_time
        iters_per_sec = len(train_loader.dataset) / total_time
        print(f"  Iterations/sec: {iters_per_sec:.2f}, Total epoch time: {total_time:.2f}s")
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation phase
        avg_val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        
        # Record history
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(avg_val_loss)
        
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
            }, "best_cnn_model.pt")
            print(f"  -> New best model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load("best_cnn_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\nFinal Results:")
    print(f"Best Val Loss: {max(val_loss_hist):.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    # Plot training progress
    plot_training_progress(train_loss_hist, train_acc_hist, val_loss_hist, 'cnn_training_progress.png')
    
    # Generate feature map visualization
    if len(test_loader) > 0:
        plot_feature_maps(model, test_loader, device, 'cnn_feature_maps.png')
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()