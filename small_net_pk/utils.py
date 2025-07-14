import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class CosmicWebDataset(Dataset):
    """Dataset class for CDM/WDM cosmic web classification"""
    
    def __init__(self, cdm_data, wdm_data, indices, transform=None):
        """
        Args:
            cdm_data: numpy array of shape [N, H, W] - CDM samples
            wdm_data: numpy array of shape [N, H, W] - WDM samples  
            indices: list of indices to use from the datasets
            transform: optional transform to apply to samples
        """
        self.cdm_data = cdm_data
        self.wdm_data = wdm_data
        self.indices = indices
        self.transform = transform
        
        # Create labels: 0 for CDM, 1 for WDM
        # We'll alternate between CDM and WDM samples
        self.samples = []
        self.labels = []
        
        for idx in indices:
            if idx < len(cdm_data):
                self.samples.append(('cdm', idx))
                self.labels.append(0.0)  # CDM = 0
                # self.labels.append(0.05)  # CDM = 0.05 for balanced dataset
            if idx < len(wdm_data):
                self.samples.append(('wdm', idx))
                self.labels.append(1.0)  # WDM = 1
                # self.labels.append(0.95)  # WDM = 0.95 for balanced dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_type, data_idx = self.samples[idx]
        label = self.labels[idx]
        
        # Get the appropriate sample
        if data_type == 'cdm':
            image = self.cdm_data[data_idx]
        else:  # wdm
            image = self.wdm_data[data_idx]
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Ensure image has shape [H, W] (remove extra dimensions if needed)
        if image.dim() > 2:
            image = image.squeeze()
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)



def load_dataset(indices, transform=None, cdm_file='cdm_data.npy', wdm_file='wdm_data.npy'):
    """
    Load CDM and WDM datasets from .npy files and create a PyTorch dataset
    
    Args:
        indices: list of indices to use for sampling
        transform: optional transform to apply to samples
        cdm_file: path to CDM .npy file
        wdm_file: path to WDM .npy file
    
    Returns:
        CosmicWebDataset: PyTorch dataset containing CDM and WDM samples
    """
    try:
        # Load the data files
        print(f"Loading CDM data from {cdm_file}...")
        cdm_data = np.load(cdm_file)
        print(f"CDM data shape: {cdm_data.shape}")
        
        print(f"Loading WDM data from {wdm_file}...")
        wdm_data = np.load(wdm_file)
        print(f"WDM data shape: {wdm_data.shape}")
        
        # Validate data shapes
        if len(cdm_data.shape) != 3 or len(wdm_data.shape) != 3:
            raise ValueError("Data should have shape [N, H, W]")
        
        # Create and return dataset
        dataset = CosmicWebDataset(cdm_data, wdm_data, indices, transform)
        print(f"Created dataset with {len(dataset)} samples")
        
        return dataset
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

  
def plot_training_progress(train_loss, train_acc, val_acc, save_path='training_progress_cnn.png'):
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
    ax2.plot(epochs, train_acc, label="Train Accuracy", color='tab:red', linestyle='--')
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color='tab:orange', linestyle='--', marker='o')
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


def plot_feature_maps(model, data_loader, device, save_path='feature_maps_cnn.png'):
    """Visualize feature maps from the first convolutional layer"""
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images[:1].to(device)  # Take first image
            
            # Get feature maps from first conv layer
            if hasattr(model, 'features'):
                # For SimpleCNN
                x = images
                for i in range(3):  # First conv, BN, ReLU
                    x = model.features[i](x)
                # First conv layer
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

def extract_features(model, dataloader, device):
    """Extract flattened CNN features for linear classifier"""
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Manual forward through CNN backbone
            x = model.stem(imgs)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = torch.flatten(x, 1)

            features.append(x.cpu())
            labels.append(lbls.cpu())

    return torch.cat(features), torch.cat(labels)
