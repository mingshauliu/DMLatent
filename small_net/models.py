import torch
import torch.nn as nn
from torchvision.models import resnet18



class WDMClassifierDilatedResNet(nn.Module):
    def __init__(self, num_classes=1, dropout=0.2):
        super().__init__()
        base = resnet18(weights=None)

        # Update for 1-channel input
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Disable stride in later blocks
        base.layer3[0].conv1.stride = (1, 1)
        base.layer3[0].downsample[0].stride = (1, 1)
        base.layer4[0].conv1.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)

        # Add dilation
        for name, m in base.layer3.named_modules():
            if "conv2" in name:
                m.dilation = (2, 2)
                m.padding = (2, 2)
        for name, m in base.layer4.named_modules():
            if "conv2" in name:
                m.dilation = (4, 4)
                m.padding = (4, 4)

        # Save blocks
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global pool
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)      # [B, 64, H/4, W/4]
        x = self.layer1(x)    # [B, 64, ...]
        x = self.layer2(x)    # [B, 128, ...]
        x = self.layer3(x)    # [B, 256, ...]
        x = self.layer4(x)    # [B, 512, H', W']
        x = self.classifier(x)
        return x


# class WDMClassifierTiny(nn.Module):
#     """Lightweight CNN for CDM/WDM classification"""
#     def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # 256x256
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 128x128

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 64x64

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 32x32

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 16x16
#         )

#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1]
#             nn.Flatten(),
#             nn.Dropout(dropout),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


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

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
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

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
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

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 1024, 1, 1]
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
class WDMClassifierLargeCircular(nn.Module):
    """Large CNN for filament classification with deeper architecture"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='circular'), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='circular'), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='circular'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='circular'),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # check this
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x