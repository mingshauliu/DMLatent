import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Upsample → ConvBlock with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class WDMUNetClassifier(nn.Module):
    """U-Net with global classification and exposed intermediate features"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, 32)   # 128x128
        self.enc2 = ConvBlock(32, 64)           # 128x128
        self.enc3 = ConvBlock(64, 128)          # 32x32
        self.enc4 = ConvBlock(128, 256)          # 32x32

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(256, 512)   # 16x16

        # --- Decoder ---
        self.dec4 = UpBlock(512, 256)           # 32x32
        self.dec3 = UpBlock(256, 128)            # 32x32
        self.dec2 = UpBlock(128, 64)            # 128x128
        self.dec1 = UpBlock(64, 32)             # 128x128

        # --- Classification ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + 32, num_classes)
        )

        self.down = nn.MaxPool2d(2)

    def forward(self, x, return_features=False):
        # --- Encoder ---
        x1 = self.enc1(x)                 # [B, 64, 128, 128]
        x2 = self.enc2(self.down(x1))     # [B, 64, 64, 64]
        x3 = self.enc3(self.down(x2))     # [B, 128, 64, 64]
        x4 = self.enc4(self.down(x3))     # [B, 256, 32, 32]

        # --- Bottleneck ---
        x_bottleneck = self.bottleneck(self.down(x4))  # [B, 512, 16, 16]

        # --- Decoder ---
        x = self.dec4(x_bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)  # [B, 64, 128, 128]

        # --- Global Pooling ---
        pooled_bottleneck = self.pool(x_bottleneck).flatten(1)  # [B, 512]
        pooled_dec1 = self.pool(x).flatten(1)                   # [B, 64]
        features = torch.cat([pooled_bottleneck, pooled_dec1], dim=1)  # [B, 1088]

        out = self.classifier(features)  # [B, num_classes]

        if return_features:
            return out, {
                "x1": x1,  # encoder 64
                "x2": x2,  # encoder 64
                "x3": x3,  # encoder 128
                "x4": x4,  # encoder 256
                "bottleneck": x_bottleneck,  # 512
                "pooled_features": features  # final input to classifier
            }
        else:
            return out

class WDMUNetClassifierSmaller(nn.Module):
    """U-Net with global classification and exposed intermediate features"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, 16)   
        self.enc2 = ConvBlock(16, 32)           
        self.enc3 = ConvBlock(32, 64)        
        self.enc4 = ConvBlock(64, 128)        

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(128, 256)   

        # --- Decoder ---
        self.dec4 = UpBlock(256, 128)         
        self.dec3 = UpBlock(128, 64)          
        self.dec2 = UpBlock(64, 32)            
        self.dec1 = UpBlock(32, 16)             

        # --- Classification ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + 16, num_classes)
        )

        self.down = nn.MaxPool2d(2)

    def forward(self, x, return_features=False):
        # --- Encoder ---
        x1 = self.enc1(x)                
        x2 = self.enc2(self.down(x1))    
        x3 = self.enc3(self.down(x2))    
        x4 = self.enc4(self.down(x3))    

        # --- Bottleneck ---
        x_bottleneck = self.bottleneck(self.down(x4))  # [B, 512, 16, 16]

        # --- Decoder ---
        x = self.dec4(x_bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)  # [B, 64, 128, 128]

        # --- Global Pooling ---
        pooled_bottleneck = self.pool(x_bottleneck).flatten(1)  # [B, 512]
        pooled_dec1 = self.pool(x).flatten(1)                   # [B, 64]
        features = torch.cat([pooled_bottleneck, pooled_dec1], dim=1)  # [B, 1088]

        out = self.classifier(features)  # [B, num_classes]

        if return_features:
            return out, {
                "x1": x1,  # encoder 64
                "x2": x2,  # encoder 64
                "x3": x3,  # encoder 128
                "x4": x4,  # encoder 256
                "bottleneck": x_bottleneck,  # 512
                "pooled_features": features  # final input to classifier
            }
        else:
            return out

class WDMUNetClassifierBigger(nn.Module):
    """U-Net with global classification and exposed intermediate features"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, 64)   # 128x128
        self.enc2 = ConvBlock(64, 128)           # 128x128
        self.enc3 = ConvBlock(128, 256)          # 32x32
        self.enc4 = ConvBlock(256, 512)          # 32x32

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(512, 1024)   # 16x16

        # --- Decoder ---
        self.dec4 = UpBlock(1024, 512)           # 32x32
        self.dec3 = UpBlock(512, 256)            # 32x32
        self.dec2 = UpBlock(256, 128)            # 128x128
        self.dec1 = UpBlock(128, 64)             # 128x128

        # --- Classification ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024 + 64, num_classes)
        )

        self.down = nn.MaxPool2d(2)

    def forward(self, x, return_features=False):
        # --- Encoder ---
        x1 = self.enc1(x)                 # [B, 64, 128, 128]
        x2 = self.enc2(self.down(x1))     # [B, 64, 64, 64]
        x3 = self.enc3(self.down(x2))     # [B, 128, 64, 64]
        x4 = self.enc4(self.down(x3))     # [B, 256, 32, 32]

        # --- Bottleneck ---
        x_bottleneck = self.bottleneck(self.down(x4))  # [B, 512, 16, 16]

        # --- Decoder ---
        x = self.dec4(x_bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)  # [B, 64, 128, 128]

        # --- Global Pooling ---
        pooled_bottleneck = self.pool(x_bottleneck).flatten(1)  # [B, 512]
        pooled_dec1 = self.pool(x).flatten(1)                   # [B, 64]
        features = torch.cat([pooled_bottleneck, pooled_dec1], dim=1)  # [B, 1088]

        out = self.classifier(features)  # [B, num_classes]

        if return_features:
            return out, {
                "x1": x1,  # encoder 64
                "x2": x2,  # encoder 64
                "x3": x3,  # encoder 128
                "x4": x4,  # encoder 256
                "bottleneck": x_bottleneck,  # 512
                "pooled_features": features  # final input to classifier
            }
        else:
            return out