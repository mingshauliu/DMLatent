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

class UNetScalarField(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, out_channels=2):
        super().__init__()

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
        
        condition_dim = 64 + 64 + 64  # time + mass_condition + astro_condition
        
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
            
    def forward(self, x, combined_condition, total_mass_condition):
        t = combined_condition[:,0]
        params = combined_condition[:,1:]
        
        # Simple time embedding - just linear projection
        time_embed = self.time_proj(t.view(-1, 1))  # (batch, 64)
        condition_embed = self.condition_encoder(total_mass_condition)  # (batch, 64)
        param_embed = self.param_encoder(params)
        
        combined_embed = torch.cat([time_embed, param_embed, condition_embed], dim=1)  # (batch, 192)
        
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