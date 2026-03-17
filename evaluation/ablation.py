"""
Ablation Study
==============
Compares Null Tensor encoding vs Zero-Padding encoding.

From the paper:
    "Experimental results show that the proposed Null Tensor approach
     preserved the positional consistency of input features and hence
     yielded an approximately 10% increase in recall for deletion-specific
     off-targets compared with zero-padding."

This module:
1. Trains two identical models — one with each encoding
2. Evaluates both on the same test set
3. Reports the recall difference (should show ~10% gain)
"""

import os
import torch
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config, Config
from data.encoding import create_dataloaders
from evaluation.evaluate import evaluate_model
from evaluation.metrics import print_comparison_table


def run_ablation_study(
    df: pd.DataFrame,
    model_class,
    model_kwargs: dict = None,
    epochs: int = 20,
    device=None,
):
    """
    Run the encoding ablation study.

    Args:
        df: Full dataset DataFrame
        model_class: Model class to instantiate (e.g., NeuroCRISPRKAN)
        model_kwargs: Additional kwargs for model constructor
        epochs: Training epochs per variant
        device: torch device

    Returns:
        Dict with results for both encodings
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_kwargs is None:
        model_kwargs = {}

    results = {}

    for encoding_name in ["null_tensor", "zero_pad"]:
        print(f"\n{'='*60}")
        print(f"  ABLATION: Training with {encoding_name.upper()} encoding")
        print(f"{'='*60}")

        # Create dataloaders with this encoding
        loaders = create_dataloaders(
            df,
            encoder=encoding_name,
            batch_size=config.training.batch_size,
        )

        # Build fresh model
        # Note: For zero_pad encoding, CNN input channels = 4 instead of 5
        if encoding_name == "zero_pad":
            # Need to adjust CNN input channels
            from dataclasses import replace
            cfg = Config()
            cfg.cnn = replace(cfg.cnn, input_channels=4)
            model = model_class(cfg=cfg, **model_kwargs)
        else:
            model = model_class(**model_kwargs)

        model = model.to(device)

        # Train (import here to avoid circular)
        from training.train import train
        from dataclasses import replace as dr
        train_cfg = dr(config.training, epochs=epochs)

        history = train(
            model, loaders["train"], loaders["val"],
            cfg=train_cfg, device=device
        )

        # Evaluate on test set
        eval_results = evaluate_model(model, loaders["test"], device)
        results[encoding_name] = {
            "metrics": eval_results["classification"],
            "deletion_analysis": eval_results["deletion_analysis"],
            "history": history,
        }

    # Print comparison
    comparison = {
        "Null Tensor": results["null_tensor"]["metrics"],
        "Zero-Padding": results["zero_pad"]["metrics"],
    }
    print_comparison_table(comparison)

    # Highlight the key finding
    nt_recall = results["null_tensor"]["deletion_analysis"]["deletion_specific"].get("recall", 0)
    zp_recall = results["zero_pad"]["deletion_analysis"]["deletion_specific"].get("recall", 0)
    gain = (nt_recall - zp_recall) * 100

    print(f"\n{'='*60}")
    print(f"  KEY FINDING: Deletion-Specific Recall")
    print(f"{'='*60}")
    print(f"  Null Tensor:  {nt_recall:.4f}")
    print(f"  Zero-Padding: {zp_recall:.4f}")
    print(f"  Gain:         {gain:+.1f}%")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    print("Ablation study module ready.")
    print("Usage:")
    print("  from evaluation.ablation import run_ablation_study")
    print("  results = run_ablation_study(df, NeuroCRISPRKAN)")
