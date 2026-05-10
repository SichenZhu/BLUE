"""Dual-head UNet for deconvolution (ported from ori_deconv_algo.py).

- feature1 (OneLinear): n_in_gene -> 4096 — GEP latent.
- feature2 (OneLinear): n_in_gene -> 4096 — prop latent (separate on purpose).
- UNet 1-D on top of feature1: Conv1d with channels 1→16→32→64→128→256 down,
  mirrored up, final OutChannels = n_classes, then LinearRelu -> n_out_gene.
- prop head MLP on feature2: 4096→1024→256→64→n_classes (SiLU), softmax.

Expects input length = 4096 (set by feature1) which supports 4 down/up steps.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.one_linear = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.one_linear(x)


class LinearRelu(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.linear_relu(x)


class DoubleConv(nn.Module):
    """(Conv1d -> BN -> ReLU) × 2, kernel=3 padding=1."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Prop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.SiLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class UNet(nn.Module):
    def __init__(self, n_in_gene: int, n_out_gene: int, n_classes: int):
        super().__init__()
        self.n_in_gene = n_in_gene
        self.n_out_gene = n_out_gene
        self.n_classes = n_classes

        self.feature1 = OneLinear(n_in_gene, 4096)
        self.feature2 = OneLinear(n_in_gene, 4096)

        self.inc = DoubleConv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, n_classes)

        self.outGEP = LinearRelu(4096, n_out_gene)
        self.prop = Prop(4096, n_classes)

    def forward(self, x):
        feat1 = self.feature1(x)                 # (B, 4096)
        x1 = self.inc(feat1.unsqueeze(1))        # (B, 16, 4096)
        x2 = self.down1(x1)                      # (B, 32, 2048)
        x3 = self.down2(x2)                      # (B, 64, 1024)
        x4 = self.down3(x3)                      # (B, 128, 512)
        x5 = self.down4(x4)                      # (B, 256, 256)

        x_ = self.up1(x5, x4)                    # (B, 128, 512)
        x_ = self.up2(x_, x3)                    # (B, 64, 1024)
        x_ = self.up3(x_, x2)                    # (B, 32, 2048)
        x_ = self.up4(x_, x1)                    # (B, n_classes, 4096)
        recon = self.outGEP(x_)                  # (B, n_classes, n_out_gene)

        feat2 = self.feature2(x)
        prop = self.prop(feat2)                  # (B, n_classes), softmax
        return prop, recon
