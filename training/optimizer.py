"""
Optimizer & Scheduler
=====================
Adam optimizer with cosine annealing learning rate schedule.

From paper:
- Adam optimizer (no weight decay linked to it)
- Cosine annealing scheduler
- Early stopping when validation plateaus
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config


def create_optimizer(model, cfg=None):
    """
    Create Adam optimizer with separate learning rates for different components.

    DNABERT-2 (LoRA params) gets a lower learning rate than CNN and KAN.
    """
    if cfg is None:
        cfg = config.training

    # Assign each parameter to exactly one group (exclusive grouping)
    transformer_params = []
    cnn_params = []
    kan_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "transformer_stream" in name:
            transformer_params.append(param)
        elif "cnn_stream" in name:
            cnn_params.append(param)
        elif "kan_core" in name:
            kan_params.append(param)
        else:
            # fusion, chromatin_proj, and any other trainable params
            other_params.append(param)

    param_groups = []
    if transformer_params:
        param_groups.append({
            "params": transformer_params,
            "lr": cfg.learning_rate * 0.1,
            "name": "transformer_lora",
        })
    if cnn_params:
        param_groups.append({
            "params": cnn_params,
            "lr": cfg.learning_rate,
            "name": "cnn",
        })
    if kan_params:
        param_groups.append({
            "params": kan_params,
            "lr": cfg.learning_rate,
            "name": "kan",
        })
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": cfg.learning_rate,
            "name": "other",
        })

    optimizer = Adam(
        param_groups,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    return optimizer


def create_scheduler(optimizer, cfg=None):
    """Create cosine annealing learning rate scheduler."""
    if cfg is None:
        cfg = config.training

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-7,
    )

    return scheduler


class EarlyStopping:
    """
    Early stopping to halt training when validation loss plateaus.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop
