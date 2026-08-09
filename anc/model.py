import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        dilation=1,
    ):
        super().__init__()

        self.left_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.norm = nn.GroupNorm(
            1,
            out_channels,
        )

        self.activation = nn.PReLU(
            out_channels
        )

    def forward(self, x):
        x = F.pad(
            x,
            (self.left_padding, 0),
        )

        x = self.conv(x)
        x = self.norm(x)

        return self.activation(x)


class SHAANCNet(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 64

        self.input_block = CausalConvBlock(
            1,
            channels,
            kernel_size=7,
            dilation=1,
        )

        self.blocks = nn.Sequential(
            CausalConvBlock(
                channels,
                channels,
                dilation=1,
            ),
            CausalConvBlock(
                channels,
                channels,
                dilation=2,
            ),
            CausalConvBlock(
                channels,
                channels,
                dilation=4,
            ),
            CausalConvBlock(
                channels,
                channels,
                dilation=8,
            ),
            CausalConvBlock(
                channels,
                channels,
                dilation=16,
            ),
            CausalConvBlock(
                channels,
                channels,
                dilation=32,
            ),
            CausalConvBlock(
                channels,
                channels,
                dilation=64,
            ),
        )

        self.output_layer = nn.Conv1d(
            channels,
            1,
            kernel_size=1,
        )

    def forward(self, noisy):
        features = self.input_block(noisy)

        residual = features

        features = self.blocks(features)
        features = features + residual

        estimated_noise = self.output_layer(
            features
        )

        enhanced = noisy - estimated_noise

        return torch.clamp(
            enhanced,
            -1.0,
            1.0,
        )


def create_model():
    return SHAANCNet()