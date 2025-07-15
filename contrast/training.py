import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
from models import ContrastiveCNN, NECTLoss, WDMClassifierTiny
from utils import CDMWDMPairDataset, load_contrastive_dataset
from transform import TensorAugment


# ------------------ Training Script ------------------
def train():
    config = {
        'img_size': 256,
        'dropout': 0.1,  # Dropout rate
        'batch_size': 64,
        'lr': 5e-5,  # Learning rate
        'weight_decay': 1e-4,
        'epochs': 80,
        'patience': 20,  # Early stopping patience
        'k_samples': 15000,  # Number of samples to use
        'model_type': 'tiny',  # 'simple' or 'big'
        'blur_kernel': 0,
        # 'conv_kernel_size': 'adj',  # Kernel size for convolutional layers
        'conv_kernel_size': 9,  # Kernel size for convolutional layers
        'normalize': False  # Normalize images to Gaussian
    }
    cdm_file='/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy'
    wdm_file='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mtot_IllustrisTNG_WDM_z=0.00.npy'
    print("=== Contrastive Learning Training ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = TensorAugment(
        size=(config['img_size'], config['img_size']),
        p_flip=0.5, # No flipping and rotation for now
        p_rot=0.5,
        noise_std=0, # Stop noise temporarily
        blur_kernel=config['blur_kernel'],
        apply_log=True,  # Keep log scale
        normalize=config['normalize']  # Normalize images to [0, 1]
    )
     # Sample indices
    all_indices = random.sample(range(15000), config['k_samples'])
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
    
    
    train_dataset = load_contrastive_dataset(train_indices, transform=my_augment, 
                                         cdm_file='path/to/cdm.npy', 
                                         wdm_file='path/to/wdm.npy')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


    model = ContrastiveCNN(WDMClassifierTiny()).to(device)
    loss_fn = NECTLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x1, x2, y1, y2 in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = loss_fn(z1, z2, y1, y2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    train()
