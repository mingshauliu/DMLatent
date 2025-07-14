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

# ------------------ Base Encoder ------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.out_features = 32 * 16 * 16  # Assumes input 64x64

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)

# ------------------ Projection Head ------------------
class ContrastiveCNN(nn.Module):
    def __init__(self, base_cnn, projection_dim=128):
        super().__init__()
        self.encoder = base_cnn
        self.projector = nn.Sequential(
            nn.Linear(base_cnn.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        return F.normalize(self.projector(features), dim=1)

# ------------------ Contrastive Loss ------------------
class NECTLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, labels_i, labels_j):
        logits = torch.matmul(z_i, z_j.T) / self.temperature
        logits_mask = torch.eye(logits.shape[0], device=logits.device).bool()
        labels_match = (labels_i.unsqueeze(1) == labels_j.unsqueeze(0)).float()
        labels_match = labels_match.masked_fill_(logits_mask, 0)
        sim = F.softmax(logits, dim=1)
        loss = -torch.sum(labels_match * torch.log(sim + 1e-8)) / labels_match.sum()
        return loss

# ------------------ Dataset Class ------------------
class CDMWDMPairDataset(Dataset):
    def __init__(self, cdm_data, wdm_data, transform=None):
        self.cdm_data = cdm_data
        self.wdm_data = wdm_data
        self.transform = transform

    def __len__(self):
        return max(len(self.cdm_data), len(self.wdm_data))

    def __getitem__(self, idx):
        if random.random() < 0.5:
            dataset = self.cdm_data if random.random() < 0.5 else self.wdm_data
            x1 = dataset[random.randint(0, len(dataset) - 1)][0]
            x2 = dataset[random.randint(0, len(dataset) - 1)][0]
            label1 = label2 = 0 if dataset == self.cdm_data else 1
        else:
            x1 = self.cdm_data[random.randint(0, len(self.cdm_data) - 1)][0]
            x2 = self.wdm_data[random.randint(0, len(self.wdm_data) - 1)][0]
            label1, label2 = 0, 1
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x1, x2, torch.tensor(label1), torch.tensor(label2)

# ------------------ Minimal Augmentations ------------------
def get_contrastive_transform(crop_size=None):
    t_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([0, 90, 180, 270])
    ]
    if crop_size:
        t_list.insert(0, transforms.RandomCrop(crop_size))
    t_list.append(transforms.ToTensor())
    return transforms.Compose(t_list)

# ------------------ Dummy Image Data Generator ------------------
def create_dummy_dataset(n, label):
    data = []
    for _ in range(n):
        img = Image.fromarray((np.random.rand(64, 64) * 255).astype(np.uint8)).convert("L")
        data.append((img, label))
    return data

# ------------------ Training Script ------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_contrastive_transform()
    cdm_data = create_dummy_dataset(500, 0)
    wdm_data = create_dummy_dataset(500, 1)
    dataset = CDMWDMPairDataset(cdm_data, wdm_data, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ContrastiveCNN(SimpleCNN()).to(device)
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
