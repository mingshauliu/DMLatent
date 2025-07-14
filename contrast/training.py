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
from utils import get_contrastive_transform, CDMWDMPairDataset


# ------------------ Training Script ------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_contrastive_transform()
    cdm_data = create_dummy_dataset(500, 0)
    wdm_data = create_dummy_dataset(500, 1)
    dataset = CDMWDMPairDataset(cdm_data, wdm_data, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
