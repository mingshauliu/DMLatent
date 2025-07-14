import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def __init__(self, in_channels=1, pk_dim=127, num_classes=1, dropout=0.2):
        super().__init__()
        self.pk_dim = pk_dim
        
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
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 + pk_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, pk_vec=None):
        x = self.features(x)          # [B, 256, 16, 16]
        x = self.cnn_head(x)          # Global pool + flatten -> [B, 256]

        if pk_vec is not None:
            x = torch.cat([x, pk_vec], dim=1)  # [B, 256 + pk_dim]
        else:
            # During inference, fill in zeros if pk_vec is not provided
            batch_size = x.size(0)
            device = x.device
            zero_vec = torch.zeros(batch_size, self.pk_dim, device=device)
            x = torch.cat([x, zero_vec], dim=1)

        x = self.fc(x)                # [B, 1]
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