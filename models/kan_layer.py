"""
Kolmogorov-Arnold Network (KAN) Layer
======================================
Custom implementation of KAN with learnable B-spline activation functions.

Key idea from the paper:
- Standard MLPs use fixed activations (ReLU/Sigmoid) on NODES
- KAN places LEARNABLE B-spline functions on EDGES
- This gives "local plasticity" to model sharp nonlinear decision boundaries
- Better for capturing rare off-target cleavage patterns (reduces spectral bias)

Mathematical formulation:
    For input x ∈ R^n, KAN output:
        Φ(x) = Σ_{q=1}^{Q} φ_{q,p}(x_p)
    where φ_{q,p} is a learnable B-spline function parameterized by control points.

Implementation:
    - Each edge has a B-spline basis with `num_knots` knot points
    - Spline coefficients are learnable parameters
    - B-spline order is configurable (default: cubic, order=3)
    - L1 regularization on coefficients prevents overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


class BSplineBasis(nn.Module):
    """
    Compute B-spline basis functions for given input values.

    Given knot vector and order, evaluates the B-spline basis
    at each input point. This is the core math behind KAN edges.
    """

    def __init__(self, num_knots: int = 8, order: int = 3, x_range: tuple = (-1, 1)):
        super().__init__()
        self.order = order
        self.num_knots = num_knots
        self.num_bases = num_knots + order - 1

        # Create uniform knot vector with padding for boundary conditions
        # Padded knots ensure basis functions cover the full range
        knot_step = (x_range[1] - x_range[0]) / (num_knots - 1)
        knots = torch.linspace(
            x_range[0] - order * knot_step,
            x_range[1] + order * knot_step,
            num_knots + 2 * order
        )
        self.register_buffer("knots", knots)

    def forward(self, x):
        """
        Evaluate B-spline basis at points x.

        Args:
            x: Tensor of shape (..., ) — input values

        Returns:
            Tensor of shape (..., num_bases) — basis function values
        """
        x = x.unsqueeze(-1)  # (..., 1)
        knots = self.knots   # (num_knots + 2*order, )

        # Iterative Cox-de Boor recursion for B-spline evaluation
        # Order 0: piecewise constant
        bases = ((x >= knots[:-1]) & (x < knots[1:])).float()  # (..., num_knots+2*order-1)

        # Higher orders via recursion
        for k in range(1, self.order + 1):
            n = bases.size(-1) - 1  # number of basis functions at this level

            # Left term: (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
            t_left = knots[:n]
            t_left_k = knots[k:k + n]
            left_num = x - t_left
            left_den = t_left_k - t_left

            # Right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
            t_right = knots[k + 1:k + 1 + n]
            t_right_den = knots[1:1 + n]
            right_num = t_right - x
            right_den = t_right - t_right_den

            # Avoid division by zero
            left = torch.where(
                left_den.abs() > 1e-8,
                left_num / left_den,
                torch.zeros_like(left_num)
            )
            right = torch.where(
                right_den.abs() > 1e-8,
                right_num / right_den,
                torch.zeros_like(right_num)
            )

            bases = left * bases[..., :-1] + right * bases[..., 1:]

        return bases  # (..., num_bases)


class KANLayer(nn.Module):
    """
    Single KAN layer: replaces Dense layer with spline-based edges.

    Instead of: y = activation(W @ x + b)  [MLP]
    KAN does:   y_j = Σ_i φ_{i,j}(x_i)    [sum of learnable spline functions]

    Each (input, output) pair has its own B-spline function,
    parameterized by learnable coefficients.
    """

    def __init__(self, in_features: int, out_features: int, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.kan

        self.in_features = in_features
        self.out_features = out_features

        # B-spline basis (shared across all edges)
        self.basis = BSplineBasis(
            num_knots=cfg.num_knots,
            order=cfg.spline_order,
        )

        # Learnable spline coefficients for each edge
        # Shape: (in_features, out_features, num_bases)
        num_bases = cfg.num_knots + cfg.spline_order - 1
        self.spline_coeffs = nn.Parameter(
            torch.randn(in_features, out_features, num_bases) * 0.1
        )

        # Optional: residual linear connection (helps training stability)
        self.residual_weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.1
        )

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_features)

        Returns:
            Tensor of shape (batch, out_features)
        """
        batch_size = x.size(0)

        # Normalize input to [-1, 1] range for B-spline evaluation
        x_norm = torch.tanh(x)  # Soft normalization

        # Evaluate B-spline basis for each input dimension
        # x_norm: (batch, in_features) → basis: (batch, in_features, num_bases)
        bases = self.basis(x_norm)

        # Compute spline outputs: sum over basis functions
        # bases: (batch, in, num_bases), coeffs: (in, out, num_bases)
        # → spline_out: (batch, in, out)
        spline_out = torch.einsum("bin,ion->bio", bases, self.spline_coeffs)

        # Sum over input dimensions → (batch, out)
        output = spline_out.sum(dim=1)

        # Add residual linear path
        residual = x @ self.residual_weight  # (batch, out)
        output = output + residual + self.bias

        return output

    def get_l1_loss(self):
        """L1 regularization on spline coefficients (prevents overfitting)."""
        return self.spline_coeffs.abs().mean()


class KANDecisionCore(nn.Module):
    """
    Full KAN Decision Core: stacks multiple KAN layers.

    Replaces the traditional MLP classifier head.
    Input: fused feature vector from CNN + Transformer
    Output: binary off-target risk probability

    Architecture:
        KAN Layer 1: 256 → 128
        KAN Layer 2: 128 → 64
        Output: 64 → 1 (sigmoid for probability)
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.kan

        # Build KAN layers
        dims = [cfg.input_dim] + cfg.hidden_dims + [cfg.output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(KANLayer(dims[i], dims[i + 1], cfg))

        # Layer normalization between KAN layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in cfg.hidden_dims
        ])

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, input_dim) — fused features

        Returns:
            risk_logit: Tensor of shape (batch, 1) — raw logit (apply sigmoid externally)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply norm and dropout for hidden layers (not output)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = self.dropout(x)

        return x  # Raw logit — sigmoid applied in loss or inference

    def get_spline_l1_loss(self):
        """Total L1 regularization across all KAN layers."""
        total = 0
        for layer in self.layers:
            total += layer.get_l1_loss()
        return total / len(self.layers)


if __name__ == "__main__":
    # Test B-spline basis
    basis = BSplineBasis(num_knots=8, order=3)
    x = torch.linspace(-1, 1, 100)
    b = basis(x)
    print(f"B-spline basis shape: {b.shape}")  # (100, 10)

    # Test single KAN layer
    kan_layer = KANLayer(256, 128)
    dummy = torch.randn(4, 256)
    out = kan_layer(dummy)
    print(f"KAN layer output: {out.shape}")  # (4, 128)
    print(f"KAN L1 loss: {kan_layer.get_l1_loss().item():.4f}")

    # Test full decision core
    core = KANDecisionCore()
    dummy = torch.randn(4, 256)
    logit = core(dummy)
    print(f"KAN core output: {logit.shape}")  # (4, 1)
    print(f"KAN total L1: {core.get_spline_l1_loss().item():.4f}")

    # Parameter count
    total = sum(p.numel() for p in core.parameters())
    print(f"KAN core parameters: {total:,}")
