import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, dilation=1, dropout=0.2):
        super().__init__()
        if padding is None:
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class MultiScaleBlock(nn.Module):
    """Multi-scale block to capture features at different scales simultaneously"""
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        branch_channels = out_channels // 4
        
        # Different scale branches
        self.branch1 = ConvBlock(in_channels, branch_channels, kernel_size=1, dropout=dropout)
        self.branch2 = ConvBlock(in_channels, branch_channels, kernel_size=3, dropout=dropout)
        self.branch3 = ConvBlock(in_channels, branch_channels, kernel_size=5, dropout=dropout)
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, branch_channels, kernel_size=1, dropout=dropout)
        )
        
        # Fusion
        self.fusion = ConvBlock(out_channels, out_channels, kernel_size=1, dropout=dropout)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(concat)

class CosmicWebCNN(nn.Module):
    """CNN optimized for cosmic web classification with hierarchical multi-scale features"""
    
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Local fine structures (256×256 → 128×128)
            # Small kernels, no dilation - capture local DM clumps
            ConvBlock(in_channels, 32, kernel_size=3, dropout=dropout),
            ConvBlock(32, 32, kernel_size=3, dropout=dropout),
            nn.MaxPool2d(2, 2),
            
            # Block 2: Small-scale structures (128×128 → 64×64)
            # Slightly larger kernels to connect nearby structures
            ConvBlock(32, 64, kernel_size=5, dropout=dropout),
            ConvBlock(64, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool2d(2, 2),
            
            # Block 3: Intermediate structures (64×64 → 32×32)
            # Introduction of mild dilation for broader context
            ConvBlock(64, 128, kernel_size=5, dilation=2, dropout=dropout),
            ConvBlock(128, 128, kernel_size=3, dilation=2, dropout=dropout),
            nn.MaxPool2d(2, 2),
            
            # Block 4: Large-scale structures (32×32 → 16×16)
            # Multi-scale processing for filaments and voids
            MultiScaleBlock(128, 256, dropout=dropout),
            ConvBlock(256, 256, kernel_size=7, dilation=3, dropout=dropout),
            nn.MaxPool2d(2, 2),
            
            # Block 5: Global web patterns (16×16 → 8×8)
            # High dilation for global connectivity patterns
            ConvBlock(256, 512, kernel_size=5, dilation=4, dropout=dropout),
            ConvBlock(512, 512, kernel_size=3, dilation=6, dropout=dropout),
            nn.MaxPool2d(2, 2)
        )
        
        # Alternative: Atrous Spatial Pyramid Pooling for final global features
        self.aspp = AtrousSpatialPyramidPooling(512, 512, [1, 2, 4, 8])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return x

class AtrousSpatialPyramidPooling(nn.Module):
    """ASPP for capturing multi-scale context in final layers"""
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                self.branches.append(
                    ConvBlock(in_channels, out_channels//4, kernel_size=1)
                )
            else:
                self.branches.append(
                    ConvBlock(in_channels, out_channels//4, kernel_size=3, dilation=dilation)
                )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(in_channels, out_channels//4, kernel_size=1)
        )
        
        self.fusion = ConvBlock(out_channels + out_channels//4, out_channels, kernel_size=1)
    
    def forward(self, x):
        size = x.shape[-2:]
        
        # Process branches
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = nn.functional.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        concat = torch.cat(branch_outputs + [global_feat], dim=1)
        return self.fusion(concat)

# Alternative simpler version focusing on systematic dilation progression
class SimplifiedCosmicWebCNN(nn.Module):
    """Simplified version with systematic dilation progression"""
    
    def __init__(self, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Local features (no dilation)
            ConvBlock(in_channels, 32, kernel_size=3, dropout=dropout),
            ConvBlock(32, 32, kernel_size=3, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 256→128
            
            # Block 2: Small-scale connections
            ConvBlock(32, 64, kernel_size=3, dropout=dropout),
            ConvBlock(64, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 128→64
            
            # Block 3: Intermediate structures (mild dilation)
            ConvBlock(64, 128, kernel_size=3, dilation=2, dropout=dropout),
            ConvBlock(128, 128, kernel_size=5, dilation=2, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 64→32
            
            # Block 4: Large structures (moderate dilation)
            ConvBlock(128, 256, kernel_size=3, dilation=4, dropout=dropout),
            ConvBlock(256, 256, kernel_size=5, dilation=4, dropout=dropout),
            nn.MaxPool2d(2, 2),  # 32→16
            
            # Block 5: Global patterns (high dilation)
            ConvBlock(256, 512, kernel_size=3, dilation=8, dropout=dropout),
            ConvBlock(512, 512, kernel_size=3, dilation=8, dropout=dropout),
            nn.MaxPool2d(2, 2)   # 16→8
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x