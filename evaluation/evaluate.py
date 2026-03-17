"""
Evaluation Pipeline
===================
Runs the trained model on the test set and computes all metrics.
Generates the results needed for Table 1 in the paper.
"""

import os
import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config
from evaluation.metrics import (
    compute_all_metrics, compute_spearman_correlation,
    compute_deletion_specific_metrics, print_metrics_table,
    print_comparison_table
)
from utils.helpers import load_checkpoint


@torch.no_grad()
def evaluate_model(model, test_loader, device=None):
    """
    Run full evaluation on test set.

    Args:
        model: Trained NeuroCRISPRKAN model
        test_loader: Test DataLoader
        device: torch device

    Returns:
        Dict with all metrics + raw predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_efficiencies_true = []
    all_deletions = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        encoded = batch["encoded"].to(device)
        labels = batch["label"]
        sgrna_seqs = batch["sgrna_seq"]
        dna_seqs = batch["dna_seq"]
        efficiencies = batch["efficiency"]
        deletions = batch["has_deletion"]

        outputs = model(encoded, sgrna_seqs, dna_seqs, device=device)
        probs = outputs["risk_prob"].cpu().numpy()

        all_preds.extend(probs.flatten())
        all_labels.extend(labels.numpy().flatten())
        all_efficiencies_true.extend(efficiencies.numpy().flatten())
        all_deletions.extend(deletions.numpy().flatten())

    # Compute all metrics
    classification_metrics = compute_all_metrics(all_labels, all_preds)

    # Spearman correlation (using risk scores as proxy for efficiency)
    spearman = compute_spearman_correlation(all_efficiencies_true, all_preds)

    # Deletion-specific analysis
    deletion_analysis = compute_deletion_specific_metrics(
        all_labels, all_preds, all_deletions
    )

    # Print results
    print_metrics_table(classification_metrics, "Neuro-CRISPR-KAN Test Results")
    print(f"Spearman ρ: {spearman['spearman_rho']:.4f} "
          f"(p={spearman['spearman_p_value']:.2e})")
    print(f"\nDeletion-specific recall: "
          f"{deletion_analysis['deletion_specific'].get('recall', 'N/A')}")
    print(f"Non-deletion recall: "
          f"{deletion_analysis['non_deletion'].get('recall', 'N/A')}")

    return {
        "classification": classification_metrics,
        "spearman": spearman,
        "deletion_analysis": deletion_analysis,
        "raw": {
            "predictions": all_preds,
            "labels": all_labels,
            "efficiencies": all_efficiencies_true,
            "deletions": all_deletions,
        }
    }


def evaluate_baselines(test_loader, models_dict, device=None):
    """
    Evaluate multiple models for comparison (Table 1 reproduction).

    Args:
        test_loader: Test DataLoader
        models_dict: {"ModelName": model_instance, ...}
        device: torch device

    Returns:
        Comparison dict for print_comparison_table
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        eval_results = evaluate_model(model, test_loader, device)
        results[name] = eval_results["classification"]

    print_comparison_table(results)
    return results


if __name__ == "__main__":
    print("Evaluation module ready.")
    print("Usage:")
    print("  from evaluation.evaluate import evaluate_model")
    print("  results = evaluate_model(model, test_loader)")
