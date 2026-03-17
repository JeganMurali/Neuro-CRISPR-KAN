"""
Run Training Script
===================
Single-entry-point to train the full Neuro-CRISPR-KAN model.
Handles: dataset generation → model building → training → evaluation.
"""

import os
import sys
import io

# Fix Windows console Unicode (cp1252 → utf-8)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import config
from utils.helpers import set_seed, get_device, count_parameters
from data.data_generation import generate_dataset, save_dataset
from data.encoding import create_dataloaders


def main():
    # ============================
    # 1. Setup
    # ============================
    set_seed(config.seed)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  NEURO-CRISPR-KAN TRAINING PIPELINE")
    print(f"{'='*60}")

    # ============================
    # 2. Generate Dataset
    # ============================
    print("\n[Step 1/4] Generating synthetic dataset...")
    df = generate_dataset()
    save_dataset(df)

    # ============================
    # 3. Create DataLoaders
    # ============================
    print("\n[Step 2/4] Creating DataLoaders with Null Tensor encoding...")
    loaders = create_dataloaders(
        df,
        encoder="null_tensor",
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )

    # ============================
    # 4. Build Model
    # ============================
    print("\n[Step 3/4] Building Neuro-CRISPR-KAN model...")
    print("  -> Loading DNABERT-2 (first time downloads ~500MB)...")
    print("  -> Applying LoRA adapters (rank=8, alpha=16)...")

    from models.neuro_crispr_kan import NeuroCRISPRKAN
    model = NeuroCRISPRKAN(config)

    params = count_parameters(model)
    print(f"\n  Model Parameters:")
    print(f"    Total:     {params['total']:,}")
    print(f"    Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    print(f"    Frozen:    {params['frozen']:,}")

    # Check VRAM usage estimate
    param_mem = params['total'] * 4 / (1024**3)  # 4 bytes per float32
    print(f"    Est. memory: ~{param_mem:.1f} GB (params only)")

    # ============================
    # 5. Train
    # ============================
    print(f"\n[Step 4/4] Starting training on {device}...")
    from training.train import train
    history = train(model, loaders["train"], loaders["val"], device=device)

    # ============================
    # 6. Evaluate
    # ============================
    print("\n[Bonus] Running final evaluation on test set...")
    from evaluation.evaluate import evaluate_model
    results = evaluate_model(model, loaders["test"], device)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best model saved to: {config.training.checkpoint_dir}/best_model.pt")
    print(f"  Final model saved to: {config.training.checkpoint_dir}/final_model.pt")
    print(f"{'='*60}\n")

    return history, results


if __name__ == "__main__":
    main()
