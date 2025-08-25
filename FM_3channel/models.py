import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.linear = nn.Linear(condition_dim, feature_dim * 2)
    
    def forward(self, features, condition_embed):
        scale_shift = self.linear(condition_embed)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.view(-1, features.size(1), 1, 1)
        shift = shift.view(-1, features.size(1), 1, 1)
        return features * (1 + scale) + shift

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim=128, down=True):
        super().__init__()
        self.down = down
        
        if down:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.film1 = FiLMLayer(condition_dim, out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.film2 = FiLMLayer(condition_dim, out_channels)
            self.pool = nn.MaxPool2d(2)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.film1 = FiLMLayer(condition_dim, out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.film2 = FiLMLayer(condition_dim, out_channels)
    
    def forward(self, x, condition_embed):
        x = self.conv1(x)
        x = self.film1(x, condition_embed)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.film2(x, condition_embed)
        x = F.relu(x)
        
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
    def __init__(self, in_channels=3, embedding_dim=64):
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

        # Add ResNet branch for processing two-channel input
        self.resnet_branch = ResNetBranch(in_channels=3, embedding_dim=64)

        self.upconv4 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(base_channels, base_channels//2, 2, stride=2)
        self.output = nn.Conv2d(base_channels//2, out_channels, 1)
        
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels//4, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels//4, 64)
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )

        # Simple time projection - no sinusoidal embedding
        self.time_proj = nn.Linear(1, 64)
        
        # Updated condition_dim to include ResNet embedding
        condition_dim = 64 + 64 + 64 + 64  # time + mass_condition + astro_condition + resnet_embedding
        
        self.enc1 = UNetBlock(in_channels, base_channels, condition_dim, down=True)
        self.enc2 = UNetBlock(base_channels, base_channels*2, condition_dim, down=True)
        self.enc3 = UNetBlock(base_channels*2, base_channels*4, condition_dim, down=True)
        self.enc4 = UNetBlock(base_channels*4, base_channels*8, condition_dim, down=True)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_film = FiLMLayer(condition_dim, base_channels*8)
        
        self.dec4 = UNetBlock(base_channels*8 + base_channels*4, base_channels*4, condition_dim, down=False) 
        self.dec3 = UNetBlock(base_channels*4 + base_channels*2, base_channels*2, condition_dim, down=False) 
        self.dec2 = UNetBlock(base_channels*2 + base_channels, base_channels, condition_dim, down=False)     
        self.dec1 = UNetBlock(base_channels + base_channels//2, base_channels//2, condition_dim, down=False) 
            
    def forward(self, x, combined_condition, total_mass_condition, resnet_input):
        t = combined_condition[:,0]
        params = combined_condition[:,1:]
        
        # Process ResNet input
        resnet_embed = self.resnet_branch(resnet_input)  # (batch, 64)
        
        # Simple time embedding - just linear projection
        time_embed = self.time_proj(t.view(-1, 1))  # (batch, 64)
        condition_embed = self.condition_encoder(total_mass_condition)  # (batch, 64)
        param_embed = self.param_encoder(params)
        
        # Concatenate all embeddings including ResNet output
        combined_embed = torch.cat([time_embed, param_embed, condition_embed, resnet_embed], dim=1)  # (batch, 256)
        
        # Encoder
        skip1, x = self.enc1(x, combined_embed)
        skip2, x = self.enc2(x, combined_embed)
        skip3, x = self.enc3(x, combined_embed)
        skip4, x = self.enc4(x, combined_embed)
        
        # Bottleneck with FiLM
        x = self.bottleneck(x)
        x = self.bottleneck_film(x, combined_embed)
        
        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.dec4(x, combined_embed)
        
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec3(x, combined_embed)
        
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x, combined_embed)
        
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, combined_embed)
        
        return self.output(x)
