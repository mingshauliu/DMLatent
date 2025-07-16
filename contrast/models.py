import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

class ContrastiveCNN(nn.Module):
    def __init__(self, base_cnn, projection_dim=128):
        super().__init__()
        self.encoder = base_cnn  # Base CNN for feature extraction
        
        # Projector head to map features to contrastive space
        self.projector = nn.Sequential(
            nn.Linear(base_cnn.out_features, 256),
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

class NXTentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        print(f"[INFO] Using NXTentLoss with temperature = {self.temperature}")

    def forward(self, z1, z2, labels_i, labels_j):
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # Concatenate both views
        z = F.normalize(z, dim=1)  # Normalize embeddings
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        mask = torch.eye(2 * N, device=z.device,dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)  # Mask diagonal to avoid self-comparison
        
        pos_pairs = torch.cat([
            torch.arange(N,2*N),
            torch.arange(0,N)
        ]).to(z.device)
        
        sim_pos = sim_matrix[torch.arange(2*N),pos_pairs]
        sim_total = torch.exp(sim_matrix).sum(dim=1)
        loss = -torch.log(torch.exp(sim_pos) / (sim_total))
        
        return loss.mean()

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

class WDMClassifierMedium(nn.Module):
    """Medium CNN for filament classification with more layers"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # ↓ 128x128
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),  # same size
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # ↓ 64x64
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),  # same
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.out_features = 256
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward_features(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)             # [B, 256]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier[2:](x)  # Dropout + Linear
        return x
    
class WDMClassifierLarge(nn.Module):
    """Large CNN for filament classification with deeper architecture"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # ↓ 128x128
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),  # same size
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # ↓ 64x64
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  # same
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.out_features = 512
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
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