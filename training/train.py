"""
Training Loop
=============
Full training pipeline for Neuro-CRISPR-KAN.

Features:
- Compound loss (Focal + Spline L1)
- Cosine annealing LR schedule
- Early stopping
- Gradient clipping
- Epoch-level logging
- Checkpoint saving (best model)
- Validation after each epoch
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config
from training.losses import CompoundLoss
from training.optimizer import create_optimizer, create_scheduler, EarlyStopping
from utils.helpers import save_checkpoint, setup_logging


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_focal = 0
    total_reg = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        encoded = batch["encoded"].to(device)
        labels = batch["label"].to(device)
        sgrna_seqs = batch["sgrna_seq"]  # List of strings
        dna_seqs = batch["dna_seq"]      # List of strings

        optimizer.zero_grad()

        # Forward pass
        outputs = model(encoded, sgrna_seqs, dna_seqs, device=device)
        risk_logit = outputs["risk_logit"]

        # Compute compound loss
        spline_l1 = model.get_spline_l1_loss()
        loss, components = criterion(risk_logit, labels, spline_l1)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += components["total"]
        total_focal += components["focal"]
        total_reg += components["spline_reg"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "focal": total_focal / num_batches,
        "spline_reg": total_reg / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        encoded = batch["encoded"].to(device)
        labels = batch["label"].to(device)
        sgrna_seqs = batch["sgrna_seq"]
        dna_seqs = batch["dna_seq"]

        outputs = model(encoded, sgrna_seqs, dna_seqs, device=device)
        risk_logit = outputs["risk_logit"]

        spline_l1 = model.get_spline_l1_loss()
        loss, components = criterion(risk_logit, labels, spline_l1)

        total_loss += components["total"]
        num_batches += 1

        # Collect predictions for metrics
        probs = outputs["risk_prob"].cpu().numpy()
        all_preds.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    # Quick accuracy computation
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    labels_array = np.array(all_labels).astype(int)
    accuracy = (preds_binary == labels_array).mean()

    return {
        "loss": total_loss / num_batches,
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train(model, train_loader, val_loader, cfg=None, device=None):
    """
    Full training pipeline.

    Args:
        model: NeuroCRISPRKAN model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        cfg: TrainingConfig
        device: torch device

    Returns:
        training_history: List of dicts with epoch-level metrics
    """
    if cfg is None:
        cfg = config.training
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(cfg.log_dir)
    model = model.to(device)

    # Setup
    criterion = CompoundLoss(cfg)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING NEURO-CRISPR-KAN")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Focal gamma: {cfg.focal_gamma}")
    print(f"λ_focal: {cfg.lambda_focal}, λ_reg: {cfg.lambda_reg}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - start_time

        # Logging
        log_msg = (
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_msg)
        logger.info(log_msg)

        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_focal": train_metrics["focal"],
            "train_reg": train_metrics["spline_reg"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": current_lr,
            "time": epoch_time,
        })

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                os.path.join(cfg.checkpoint_dir, "best_model.pt")
            )
            print(f"  >> New best model saved (val_loss: {best_val_loss:.4f})")

        # Early stopping check
        if early_stopping.step(val_metrics["loss"]):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_metrics["loss"],
        os.path.join(cfg.checkpoint_dir, "final_model.pt")
    )

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    print("Training module ready. Run from main notebook or script.")
    print("Usage:")
    print("  from training.train import train")
    print("  history = train(model, train_loader, val_loader)")
