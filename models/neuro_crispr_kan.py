"""
Neuro-CRISPR-KAN: Full Assembled Model
=======================================
End-to-end pipeline combining all components:

    Encoded Input → CNN Stream  ─┐
                                 ├─ Feature Fusion → KAN Decision Core → Risk Score
    Sequence Text → Transformer ─┘

This module ties everything together into a single nn.Module
that can be trained end-to-end.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config
from models.cnn_stream import CNNStream
from models.transformer_stream import TransformerStream
from models.fusion import FeatureFusion
from models.kan_layer import KANDecisionCore


class NeuroCRISPRKAN(nn.Module):
    """
    Full Neuro-CRISPR-KAN architecture.

    Inputs:
        encoded_pairs: Tensor (batch, 2, seq_len, channels) — Null Tensor encoded
        sgrna_seqs: List[str] — raw sgRNA sequences for Transformer
        dna_seqs: List[str] — raw DNA sequences for Transformer

    Outputs:
        risk_logit: (batch, 1) — off-target risk logit
        features: dict with intermediate representations for analysis
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config

        # Stream 1: 1D-CNN for local motif extraction
        self.cnn_stream = CNNStream(cfg.cnn)

        # Stream 2: DNABERT-2 + LoRA for global semantic analysis
        self.transformer_stream = TransformerStream(cfg.transformer)

        # Feature fusion
        self.fusion = FeatureFusion(
            cnn_dim=cfg.cnn.output_dim,
            transformer_dim=cfg.transformer.output_dim,
            use_gate=True,
        )

        # KAN decision core
        self.kan_core = KANDecisionCore(cfg.kan)

        # Optional: chromatin feature injection
        self.chromatin_proj = nn.Linear(1, 16)
        # Adjust KAN input if chromatin is used
        # This requires updating KAN input_dim to 256 + 16 = 272
        # For simplicity, we'll concatenate after fusion

    def forward(self, encoded_pairs, sgrna_seqs, dna_seqs, chromatin=None, device=None):
        """
        Full forward pass.

        Args:
            encoded_pairs: (batch, 2, seq_len, channels) — from NullTensorEncoder
            sgrna_seqs: List[str] — raw sequences for Transformer tokenization
            dna_seqs: List[str] — raw sequences for Transformer tokenization
            chromatin: (batch, 1) — optional chromatin accessibility scores
            device: torch device

        Returns:
            dict with:
                'risk_logit': (batch, 1) — raw logit for off-target risk
                'risk_prob': (batch, 1) — sigmoid probability
                'cnn_features': (batch, cnn_dim)
                'transformer_features': (batch, transformer_dim)
                'fused_features': (batch, fused_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        # Stream 1: CNN
        cnn_features = self.cnn_stream(encoded_pairs)  # (batch, 128)

        # Stream 2: Transformer
        transformer_features = self.transformer_stream(
            sgrna_seqs, dna_seqs, device
        )  # (batch, 128)

        # Fusion
        fused = self.fusion(cnn_features, transformer_features)  # (batch, 256)

        # KAN Decision Core
        risk_logit = self.kan_core(fused)  # (batch, 1)
        risk_prob = torch.sigmoid(risk_logit)

        return {
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
            "cnn_features": cnn_features,
            "transformer_features": transformer_features,
            "fused_features": fused,
        }

    def get_spline_l1_loss(self):
        """Get L1 regularization from KAN layers."""
        return self.kan_core.get_spline_l1_loss()

    def get_attention_weights(self, sgrna_seqs, dna_seqs, device=None):
        """Extract Transformer attention for visualization."""
        return self.transformer_stream.get_attention_weights(
            sgrna_seqs, dna_seqs, device
        )


class NeuroCRISPRKAN_CNNOnly(nn.Module):
    """
    Ablation variant: CNN + KAN only (no Transformer).
    Used to measure Transformer contribution.
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config

        self.cnn_stream = CNNStream(cfg.cnn)

        # KAN with CNN output dim only
        from dataclasses import replace
        kan_cfg = replace(cfg.kan, input_dim=cfg.cnn.output_dim)
        self.kan_core = KANDecisionCore(kan_cfg)

    def forward(self, encoded_pairs, **kwargs):
        cnn_features = self.cnn_stream(encoded_pairs)
        risk_logit = self.kan_core(cnn_features)
        return {
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
        }


class NeuroCRISPRKAN_MLPBaseline(nn.Module):
    """
    Ablation variant: CNN + Transformer + MLP (no KAN).
    Used to measure KAN contribution vs standard MLP.
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config

        self.cnn_stream = CNNStream(cfg.cnn)
        self.transformer_stream = TransformerStream(cfg.transformer)
        self.fusion = FeatureFusion(cfg.cnn.output_dim, cfg.transformer.output_dim)

        # Standard MLP instead of KAN
        fused_dim = cfg.cnn.output_dim + cfg.transformer.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, encoded_pairs, sgrna_seqs, dna_seqs, device=None, **kwargs):
        if device is None:
            device = next(self.parameters()).device
        cnn_features = self.cnn_stream(encoded_pairs)
        transformer_features = self.transformer_stream(sgrna_seqs, dna_seqs, device)
        fused = self.fusion(cnn_features, transformer_features)
        risk_logit = self.mlp(fused)
        return {
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
        }


if __name__ == "__main__":
    from utils.helpers import count_parameters

    print("Building Neuro-CRISPR-KAN...")
    model = NeuroCRISPRKAN()

    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    print(f"  Frozen: {params['frozen']:,}")

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 4
    dummy_encoded = torch.randn(batch_size, 2, 23, 5).to(device)
    dummy_sgrna = ["ATCGATCGATCGATCGATCGNGG"] * batch_size
    dummy_dna = ["ATCGATCGATCGATCGATCGNGG"] * batch_size

    with torch.no_grad():
        outputs = model(dummy_encoded, dummy_sgrna, dummy_dna, device=device)

    print(f"\nForward pass outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
