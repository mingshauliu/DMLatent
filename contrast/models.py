import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

class ContrastiveCNN(nn.Module):
    def __init__(self, base_cnn, projection_dim=128):
        super().__init__()
        self.encoder = base_cnn  # e.g., SimpleCNN or ResNet variant
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, dim=1)

class NECTLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        print(f"[INFO] Using NECTLoss with temperature = {self.temperature}")

    def forward(self, z_i, z_j, labels_i, labels_j):
        logits = torch.matmul(z_i, z_j.T) / self.temperature
        logits_mask = torch.eye(logits.shape[0], device=logits.device).bool()
        
        labels_match = (labels_i.unsqueeze(1) == labels_j.unsqueeze(0)).float()
        labels_match = labels_match.masked_fill_(logits_mask, 0)

        # Softmax over logits
        sim = F.softmax(logits, dim=1)
        loss = -torch.sum(labels_match * torch.log(sim + 1e-8)) / labels_match.sum()
        return loss

class WDMClassifierTiny(nn.Module):
    """Small CNN for filament classification without aggressive downsampling"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=1),  # 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # ↓ 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),  # same size, wider context
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # ↓ 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),  # same
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.out_features = 64
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward_features(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)             # [B, 128]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier[2:](x)  # Dropout + Linear
        return x

