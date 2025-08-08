import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import csv

class UNetBlock(nn.Module):
    """Basic U-Net building block"""
    
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.down = down
        
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.pool = nn.MaxPool2d(2)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
    
    def forward(self, x):
        if self.down:
            conv_out = self.conv(x)
            pool_out = self.pool(conv_out)
            return conv_out, pool_out
        else:
            return self.conv(x)


class UNetScalarField(nn.Module):
    """U-Net architecture for scalar field prediction"""
    
    def __init__(self, in_channels=4, base_channels=64, out_channels=2):
        super().__init__()
        
        self.enc1 = UNetBlock(in_channels, base_channels, down=True)
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
        
        self.upconv4 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec4 = UNetBlock(base_channels*8 + base_channels*4, base_channels*4, down=False) 
        
        self.upconv3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = UNetBlock(base_channels*4 + base_channels*2, base_channels*2, down=False) 
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec2 = UNetBlock(base_channels*2 + base_channels, base_channels, down=False)     
        
        self.upconv1 = nn.ConvTranspose2d(base_channels, base_channels//2, 2, stride=2)
        self.dec1 = UNetBlock(base_channels + base_channels//2, base_channels//2, down=False) 
        
        self.output = nn.Conv2d(base_channels//2, 2, 1)
    
    def forward(self, x, t, condition):
 
        t_expanded = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        input_tensor = torch.cat([x, t_expanded, condition], dim=1)
        
        # Encoder
        skip1, x = self.enc1(input_tensor)
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