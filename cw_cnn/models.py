import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification"""
    
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2, kernel_size=5):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            ConvBlock(in_channels, 32, dropout=dropout, kernel_size=kernel_size),  # 256x256 -> 128x128
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
            ConvBlock(128, 256, dropout=dropout, dilation=2),  # 32x32 -> 16x16
            ConvBlock(256, 256, dropout=dropout, dilation=2),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Fifth block
            ConvBlock(256, 512, dropout=dropout, dilation=2),
            ConvBlock(512, 512, dropout=dropout, dilation=2),
            nn.MaxPool2d(2, 2)   # 16x16 -> 8x8
        )
        
        # Global average pooling + classifier
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.cnn_head(x) # Global average pooling
        x = self.fc(x)
        return x

class AdjustedCNN(nn.Module):
    """Simple CNN for binary classification"""
    
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            ConvBlock(in_channels, 32, dropout=dropout, kernel_size=9),  # 256x256 -> 128x128
            ConvBlock(32, 32, dropout=dropout, kernel_size=9),
            nn.MaxPool2d(2, 2),  # 256x256 -> 128x128
            
            # Second block
            ConvBlock(32, 64, dropout=dropout, kernel_size=5),  # 128x128 -> 64x64
            ConvBlock(64, 64, dropout=dropout, kernel_size=5),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            
            # Third block
            ConvBlock(64, 128, dropout=dropout),  # 64x64 -> 32x32
            ConvBlock(128, 128, dropout=dropout),  # 64x64 -> 32x32,
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Fourth block
            ConvBlock(128, 256, dropout=dropout),  # 32x32 -> 16x16
            ConvBlock(256, 256, dropout=dropout),  # 32x32 -> 16x16
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Fifth block
            ConvBlock(256, 512, dropout=dropout),
            ConvBlock(512, 512, dropout=dropout),
            nn.MaxPool2d(2, 2)   # 16x16 -> 8x8
        )
        
        # Global average pooling + classifier
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.cnn_head(x) # Global average pooling
        x = self.fc(x)
        return x

class SimpleCNN_LowDownsample(nn.Module):
    """CNN for WDM/CDM classification with minimal downsampling"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            # First block (keep full resolution)
            ConvBlock(in_channels, 32, kernel_size=7, padding=3, dropout=dropout),
            ConvBlock(32, 32, kernel_size=7, padding=3, dropout=dropout),
            # No pooling

            # Second block (slight downsample)
            ConvBlock(32, 64, kernel_size=5, padding=2, dropout=dropout),
            ConvBlock(64, 64, kernel_size=5, padding=2, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 256x256 -> 128x128

            # Third block (moderate downsample)
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2, dropout=dropout),
            ConvBlock(128, 128, kernel_size=3, padding=2, dilation=2, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            # Fourth block (large receptive field, no more downsample)
            ConvBlock(128, 256, kernel_size=3, padding=4, dilation=4, dropout=dropout),
            ConvBlock(256, 256, kernel_size=3, padding=4, dilation=4, dropout=dropout),

            # Fifth block (retain spatial info, high-level features)
            ConvBlock(256, 512, kernel_size=3, padding=4, dilation=4, dropout=dropout),
            ConvBlock(512, 512, kernel_size=3, padding=4, dilation=4, dropout=dropout),
        )

        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global feature
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cnn_head(x)
        return self.fc(x)