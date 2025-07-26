import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class AttentionPooling2d(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.attention_map = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        self.out_channels = out_channels
        if out_channels is not None and out_channels != in_channels:
            self.output_proj=nn.Linear(in_channels, out_channels)
        else:
            self.output_proj = None
        

    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.attention_map(x)  # Shape: [B, 1, H, W]
        attn = attn.view(B, 1, -1)    # Flatten spatial dims: [B, 1, H*W]
        x = x.view(B, C, -1)          # Flatten spatial dims: [B, C, H*W]
        pooled = torch.bmm(x, attn.transpose(1, 2)).squeeze(-1)  # [B, C]
        
        if self.output_proj is not None:
            pooled = self.output_proj(pooled)
        return pooled

class CNNAttentionPooling(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout) 
        )

        self.attention_pooling = AttentionPooling2d(in_channels=128)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        
    def forward_features(self, x):
        x = self.features(x)  # [B, 128, H, W]
        pooled_avg = F.adaptive_avg_pool2d(x,1).flatten(1)
        pooled_attn = self.attention_pooling(x) 
        pooled = pooled_avg + pooled_attn
        return pooled  # [B, 128]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.fc(x)  
        return x

    #     self.classifier = nn.Sequential(
    #         AttentionPooling2d(in_channels=128),
    #         nn.Dropout(dropout),
    #         nn.Linear(128, num_classes)
    #     )
        
    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.classifier(x)


class WDMClassifierTiny(nn.Module):
    """Small CNN for filament classification without aggressive downsampling"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1, dilation=1),  # 256x256
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

class WDMClassifierTinywithDownSampling(nn.Module):
    """Small CNN for filament classification without aggressive downsampling"""
    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=9, stride=1, padding=1),  # 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1, dilation=1),  # 256x256
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
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),  
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
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

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
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
    
    

class ViTClassifier(nn.Module):
    """ViT with convolutional overlapping patch embedding for small-scale structure preservation"""
    def __init__(self, img_size=256, patch_size=16, in_channels=1, num_classes=1,
                 dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.3):
        super().__init__()
        self.patch_size = patch_size
        stride = patch_size // 2
        output_size = ((img_size + 2*(patch_size // 4) - patch_size) // stride) + 1
        self.num_patches = output_size * output_size
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))


        # --- Conv-based overlapping patch embedding ---
        self.proj = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size // 2,  # Overlap: 50% by default
            padding=patch_size // 4  # Center patches if needed
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (self.num_patches + 1), dim))
        self.dropout = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- Classification head ---
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape  # [B, 1, 256, 256]

        x = self.proj(x)  # [B, dim, H', W'] with overlap
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + N, dim]
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)  # [B, 1 + N, dim]
        cls_output = x[:, 0]     # global [CLS] token
        return self.mlp_head(cls_output)  # [B, num_classes]
