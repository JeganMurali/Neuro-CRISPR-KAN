"""
1D-CNN Stream
=============
Multi-scale 1D Convolutional Neural Network for local motif extraction.

Scans encoded sequences with kernels of sizes [3, 5, 7] to capture:
- PAM recognition patterns (short-range)
- Seed region mismatch signatures (medium-range)
- Local structural motifs around deletion sites

Architecture:
    Input (batch, 2*seq_len, channels)
    → Conv1D blocks with multiple kernel sizes
    → BatchNorm + ReLU + Dropout
    → Optional residual connections
    → Global Average Pooling
    → Output feature vector (batch, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


class ConvBlock(nn.Module):
    """Single convolution block with BatchNorm, ReLU, Dropout."""

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
        super().__init__()
        # Padding to maintain sequence length
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class ResidualConvBlock(nn.Module):
    """Conv block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions differ
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(F.relu(out + residual))
        return out


class CNNStream(nn.Module):
    """
    Multi-kernel 1D-CNN for local genomic motif extraction.

    Processes the flattened sgRNA-DNA pair encoding through
    parallel convolution paths with different kernel sizes,
    then fuses via concatenation and projection.
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.cnn

        self.input_channels = cfg.input_channels
        self.kernel_sizes = cfg.kernel_sizes
        self.num_filters = cfg.num_filters
        self.output_dim = cfg.output_dim

        # Input projection: (2 * channels) because we flatten sgRNA + DNA
        input_dim = 2 * cfg.input_channels  # 2 * 5 = 10

        # Parallel convolution branches (one per kernel size)
        self.branches = nn.ModuleList()
        for ks in cfg.kernel_sizes:
            if cfg.use_residual:
                branch = nn.Sequential(
                    ResidualConvBlock(input_dim, cfg.num_filters, ks, cfg.dropout),
                    ResidualConvBlock(cfg.num_filters, cfg.num_filters, ks, cfg.dropout),
                )
            else:
                branch = nn.Sequential(
                    ConvBlock(input_dim, cfg.num_filters, ks, cfg.dropout),
                    ConvBlock(cfg.num_filters, cfg.num_filters, ks, cfg.dropout),
                )
            self.branches.append(branch)

        # Fusion: concatenate all branches → project to output_dim
        total_filters = cfg.num_filters * len(cfg.kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(total_filters, cfg.output_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 2, seq_len, channels)
               where 2 = sgRNA + DNA

        Returns:
            Tensor of shape (batch, output_dim)
        """
        batch_size = x.size(0)

        # Flatten sgRNA + DNA: (batch, 2, seq_len, ch) → (batch, seq_len, 2*ch)
        x = x.permute(0, 2, 1, 3)  # (batch, seq_len, 2, ch)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, seq_len, 2*ch)

        # Conv1D expects (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, 2*ch, seq_len)

        # Run through parallel branches
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)  # (batch, num_filters, seq_len)
            # Global Average Pooling over sequence dimension
            pooled = out.mean(dim=2)  # (batch, num_filters)
            branch_outputs.append(pooled)

        # Concatenate all branch outputs
        fused = torch.cat(branch_outputs, dim=1)  # (batch, num_filters * num_kernels)

        # Project to output dimension
        features = self.fusion(fused)  # (batch, output_dim)

        return features


if __name__ == "__main__":
    model = CNNStream()
    # Simulate batch: (batch=4, 2 streams, seq_len=23, channels=5)
    dummy = torch.randn(4, 2, 23, 5)
    out = model(dummy)
    print(f"CNN output shape: {out.shape}")  # (4, 128)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CNN parameters: {total_params:,}")
