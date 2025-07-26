import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

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

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(257, num_classes)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(257,num_classes)

    def forward(self, x, sigma):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        sigma = sigma.view(-1,1).to(dtype=x.dtype)
        x = torch.cat([x,sigma],dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class WDMClassifierLarge(nn.Module):
    """Large CNN for filament classification with deeper architecture"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # check this
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(513, num_classes)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(513,num_classes)

    def forward(self, x, sigma):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        sigma = sigma.view(-1,1).to(dtype=x.dtype)
        x = torch.cat([x,sigma],dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
   
   
class WDMClassifierLarger(nn.Module):
    """Large CNN for filament classification with deeper architecture"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # check this
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(513, num_classes)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(513,num_classes)

    def forward(self, x, sigma):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        sigma = sigma.view(-1,1).to(dtype=x.dtype)
        x = torch.cat([x,sigma],dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class WDMClassifierHuge(nn.Module):
    """Large CNN for filament classification with deeper architecture"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # check this
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1025, num_classes)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1025,num_classes)

    def forward(self, x, sigma):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        sigma = sigma.view(-1,1).to(dtype=x.dtype)
        x = torch.cat([x,sigma],dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x