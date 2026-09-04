import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Efficient convolutional block with skip connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.skip = nn.Conv1d(in_channels, out_channels, 1, stride) if stride != 1 or in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.gelu(x + residual)
        return x


class EfficientSpeechEnhancer(nn.Module):
    """
    Lightweight speech enhancement model optimized for speed.
    ~2-3 million parameters (vs 10-20M for large models)
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_blocks=4):
        super().__init__()
        
        # Encoder
        self.enc_conv = nn.Conv1d(in_channels, hidden_channels, 7, padding=3)
        self.enc_bn = nn.BatchNorm1d(hidden_channels)
        
        # Convolutional blocks (reduced from 6 to 4)
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_channels, hidden_channels * 2) if i == 0 else
            ConvBlock(hidden_channels * 2, hidden_channels * 2) if i == 1 else
            ConvBlock(hidden_channels * 2, hidden_channels) if i == 2 else
            ConvBlock(hidden_channels, hidden_channels)
            for i in range(num_blocks)
        ])
        
        # Decoder
        self.dec_conv = nn.Conv1d(hidden_channels, in_channels, 7, padding=3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encoder
        x = F.gelu(self.enc_bn(self.enc_conv(x)))
        
        # Convolutional blocks
        for block in self.blocks:
            x = block(x)
        
        # Decoder
        x = self.dec_conv(x)
        
        # Residual connection (learn to predict clean from noisy)
        return x


def create_model():
    """Factory function for model creation"""
    return EfficientSpeechEnhancer(
        in_channels=1,
        hidden_channels=48,  # Reduced from 64 for speed
        num_blocks=4
    )