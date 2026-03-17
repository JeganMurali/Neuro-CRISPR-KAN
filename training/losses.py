"""
Loss Functions
==============
Compound objective from the paper:

    L_total = λ1 * L_focal + λ2 * L_reg

Where:
- L_focal: Focal Loss (handles extreme class imbalance)
    L_focal = -(1 - p_t)^γ * log(p_t)
    γ = 2.0 (focusing parameter, down-weights easy negatives)

- L_reg: L1 regularization on B-spline coefficients
    Prevents KAN splines from overfitting

Weights: λ1 = 0.7, λ2 = 0.25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in off-target detection.

    In genomic datasets, off-target events are rare (~30% positive).
    Focal loss down-weights easy-to-classify examples and focuses
    on hard misclassified ones.

    L_focal = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0)
        alpha: Class weight for positive class (default 0.75)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, 1) — raw model output (before sigmoid)
            targets: (batch, 1) — binary labels (0 or 1)

        Returns:
            Scalar focal loss
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross entropy (stable)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Apply focal weighting
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class CompoundLoss(nn.Module):
    """
    Compound objective combining focal loss with spline regularization.

    L_total = λ1 * L_focal + λ2 * L_reg

    The regularization term is computed from the model's KAN layers
    and passed in during the forward call.
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.training

        self.lambda_focal = cfg.lambda_focal
        self.lambda_reg = cfg.lambda_reg
        self.focal_loss = FocalLoss(gamma=cfg.focal_gamma)

    def forward(self, logits, targets, spline_l1_loss=None):
        """
        Args:
            logits: (batch, 1) — model output logits
            targets: (batch, 1) — binary labels
            spline_l1_loss: scalar — L1 norm of KAN spline coefficients

        Returns:
            total_loss: scalar
            loss_components: dict with individual loss values
        """
        focal = self.focal_loss(logits, targets)

        if spline_l1_loss is not None:
            reg = spline_l1_loss
        else:
            reg = torch.tensor(0.0, device=logits.device)

        total = self.lambda_focal * focal + self.lambda_reg * reg

        return total, {
            "total": total.item(),
            "focal": focal.item(),
            "spline_reg": reg.item(),
        }


if __name__ == "__main__":
    # Test focal loss
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8, 1)).float()
    loss = fl(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")

    # Test compound loss
    cl = CompoundLoss()
    spline_reg = torch.tensor(0.05)
    total, components = cl(logits, targets, spline_reg)
    print(f"Compound loss: {components}")
