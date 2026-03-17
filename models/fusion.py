"""
Feature Fusion Module
=====================
Concatenates features from CNN stream and Transformer stream
into a unified latent representation for the KAN decision core.

Paper description:
    "Feature vectors from both streams are joined in one unified
     latent representation; this vector feeds the KAN decision core."

Input:  CNN features (batch, 128) + Transformer features (batch, 128)
Output: Fused features (batch, 256)
"""

import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


class FeatureFusion(nn.Module):
    """
    Hierarchical Feature Integration.

    Concatenates local (CNN) and global (Transformer) features,
    applies layer norm and optional gated attention for
    adaptive weighting of the two streams.
    """

    def __init__(self, cnn_dim: int = None, transformer_dim: int = None, use_gate: bool = True):
        super().__init__()
        if cnn_dim is None:
            cnn_dim = config.cnn.output_dim
        if transformer_dim is None:
            transformer_dim = config.transformer.output_dim

        self.cnn_dim = cnn_dim
        self.transformer_dim = transformer_dim
        self.fused_dim = cnn_dim + transformer_dim
        self.use_gate = use_gate

        # Layer normalization on each stream before fusion
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.transformer_norm = nn.LayerNorm(transformer_dim)

        if use_gate:
            # Gated fusion: learn how much to weight each stream
            self.gate = nn.Sequential(
                nn.Linear(self.fused_dim, 2),
                nn.Softmax(dim=-1),
            )
            # Project both streams to same dim for gating
            self.cnn_proj = nn.Linear(cnn_dim, self.fused_dim // 2)
            self.transformer_proj = nn.Linear(transformer_dim, self.fused_dim // 2)

    def forward(self, cnn_features, transformer_features):
        """
        Args:
            cnn_features: (batch, cnn_dim) from CNNStream
            transformer_features: (batch, transformer_dim) from TransformerStream

        Returns:
            fused: (batch, fused_dim) unified feature vector
        """
        # Normalize each stream
        cnn_norm = self.cnn_norm(cnn_features)
        trans_norm = self.transformer_norm(transformer_features)

        if self.use_gate:
            # Concatenate for gate computation
            concat = torch.cat([cnn_norm, trans_norm], dim=-1)
            gate_weights = self.gate(concat)  # (batch, 2)

            # Project and weight
            cnn_proj = self.cnn_proj(cnn_norm)       # (batch, fused_dim//2)
            trans_proj = self.transformer_proj(trans_norm)  # (batch, fused_dim//2)

            # Apply gate weights
            weighted_cnn = cnn_proj * gate_weights[:, 0:1]
            weighted_trans = trans_proj * gate_weights[:, 1:2]

            fused = torch.cat([weighted_cnn, weighted_trans], dim=-1)
        else:
            # Simple concatenation
            fused = torch.cat([cnn_norm, trans_norm], dim=-1)

        return fused


if __name__ == "__main__":
    fusion = FeatureFusion(128, 128, use_gate=True)
    cnn_feat = torch.randn(4, 128)
    trans_feat = torch.randn(4, 128)
    fused = fusion(cnn_feat, trans_feat)
    print(f"Fused shape: {fused.shape}")  # (4, 256)
