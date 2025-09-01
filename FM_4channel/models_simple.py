import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.down = down
        
        if down:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d(2)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.down:
            return x, self.pool(x)
        else:
            return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBranch(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=16):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling and final projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = self.fc(x)
        return x

class UNetScalarField(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, out_channels=3):
        super().__init__()

        # Add ResNet branch for processing four-channel input
        self.resnet_branch = ResNetBranch(in_channels=4, embedding_dim=16)

        # Condition encoders
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels//4, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels//4, 64)
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Simple time projection
        self.time_proj = nn.Linear(1, 64)
        
        # Calculate total condition channels
        # We'll add condition information as additional channels to the input
        total_condition_dim = 64 + 64 + 64 + 16  # time + params + mass_condition + resnet
        condition_spatial_channels = 4  # We'll convert conditions to 4 spatial channels
        
        # Condition projection to spatial format
        self.condition_to_spatial = nn.Sequential(
            nn.Linear(total_condition_dim, condition_spatial_channels * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (condition_spatial_channels, 16, 16))
        )
        
        # Updated input channels: original + condition channels
        total_input_channels = in_channels + condition_spatial_channels
        
        # U-Net blocks (simplified without FiLM)
        self.enc1 = UNetBlock(total_input_channels, base_channels, down=True)
        self.enc2 = UNetBlock(base_channels, base_channels*2, down=True)
        self.enc3 = UNetBlock(base_channels*2, base_channels*4, down=True)
        self.enc4 = UNetBlock(base_channels*4, base_channels*8, down=True)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*16, 3, padding=1),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec4 = UNetBlock(base_channels*8 + base_channels*4, base_channels*4, down=False)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = UNetBlock(base_channels*4 + base_channels*2, base_channels*2, down=False)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec2 = UNetBlock(base_channels*2 + base_channels, base_channels, down=False)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels, base_channels//2, 2, stride=2)
        self.dec1 = UNetBlock(base_channels + base_channels//2, base_channels//2, down=False)
        
        self.output = nn.Conv2d(base_channels//2, out_channels, 1)
            
    def forward(self, x, combined_condition, total_mass_condition, resnet_input):
        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)
        
        t = combined_condition[:, 0]
        params = combined_condition[:, 1:]
        
        # Process ResNet input
        resnet_embed = self.resnet_branch(resnet_input)  # (batch, 16)
        
        # Process conditions
        time_embed = self.time_proj(t.view(-1, 1))  # (batch, 64)
        condition_embed = self.condition_encoder(total_mass_condition)  # (batch, 64)
        param_embed = self.param_encoder(params)  # (batch, 64)
        
        # Concatenate all embeddings
        combined_embed = torch.cat([time_embed, param_embed, condition_embed, resnet_embed], dim=1)  # (batch, 208)
        
        # Convert conditions to spatial format and resize to match input
        condition_spatial = self.condition_to_spatial(combined_embed)  # (batch, 4, 16, 16)
        condition_spatial = F.interpolate(condition_spatial, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate conditions with input
        input_with_conditions = torch.cat([x, condition_spatial], dim=1)  # (batch, in_channels + 4, H, W)
        
        # Encoder
        skip1, x = self.enc1(input_with_conditions)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x)
        
        return self.output(x)