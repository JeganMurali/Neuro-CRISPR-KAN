"""
Evaluation Metrics
==================
All metrics from the paper:
- Accuracy (Structural)
- Precision
- Recall (Sensitivity) — most important for safety
- F1-Score
- Spearman's Rank Correlation (ρ) — for efficiency score regression
- False Positive Rate (FPR)
- AUROC
- Matthews Correlation Coefficient (MCC)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    classification_report, roc_curve, precision_recall_curve
)
from scipy.stats import spearmanr


def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute all classification metrics from the paper.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred_proba: Predicted probabilities [0, 1]
        threshold: Classification threshold

    Returns:
        Dict with all metrics
    """
    y_true = np.array(y_true).astype(int)
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(y_true),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int((1 - y_true).sum()),
    }

    return metrics


def compute_spearman_correlation(y_true_efficiency, y_pred_efficiency):
    """
    Spearman's Rank Correlation for on-target efficiency scores.

    Measures monotonic relationship between predicted and
    experimentally reported cleavage activity.

    Args:
        y_true_efficiency: Ground truth efficiency scores
        y_pred_efficiency: Predicted efficiency scores

    Returns:
        rho: Spearman correlation coefficient
        p_value: Statistical significance
    """
    y_true = np.array(y_true_efficiency).flatten()
    y_pred = np.array(y_pred_efficiency).flatten()

    rho, p_value = spearmanr(y_true, y_pred)

    return {
        "spearman_rho": rho,
        "spearman_p_value": p_value,
    }


def compute_deletion_specific_metrics(y_true, y_pred_proba, has_deletion, threshold=0.5):
    """
    Compute metrics separately for deletion vs non-deletion samples.
    This is critical for measuring the Null Tensor encoding advantage.

    Args:
        y_true: Binary labels
        y_pred_proba: Predicted probabilities
        has_deletion: Binary flag for ΔF508 deletion presence

    Returns:
        Dict with overall + deletion-specific metrics
    """
    y_true = np.array(y_true).astype(int)
    y_pred_proba = np.array(y_pred_proba).flatten()
    has_deletion = np.array(has_deletion).astype(bool)

    # Overall metrics
    overall = compute_all_metrics(y_true, y_pred_proba, threshold)

    # Deletion-specific
    del_mask = has_deletion
    nondel_mask = ~has_deletion

    deletion_metrics = {}
    if del_mask.sum() > 0:
        deletion_metrics = compute_all_metrics(
            y_true[del_mask], y_pred_proba[del_mask], threshold
        )
    
    non_deletion_metrics = {}
    if nondel_mask.sum() > 0:
        non_deletion_metrics = compute_all_metrics(
            y_true[nondel_mask], y_pred_proba[nondel_mask], threshold
        )

    return {
        "overall": overall,
        "deletion_specific": deletion_metrics,
        "non_deletion": non_deletion_metrics,
        "deletion_recall_gain": (
            deletion_metrics.get("recall", 0) - non_deletion_metrics.get("recall", 0)
        ),
    }


def get_roc_curve_data(y_true, y_pred_proba):
    """Get ROC curve data for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def get_pr_curve_data(y_true, y_pred_proba):
    """Get Precision-Recall curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def print_metrics_table(metrics_dict, title="Evaluation Results"):
    """Pretty print metrics as a table."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

    key_metrics = ["accuracy", "precision", "recall", "f1_score", "specificity",
                   "fpr", "mcc", "auroc"]

    for key in key_metrics:
        if key in metrics_dict:
            print(f"  {key:<20s}: {metrics_dict[key]:.4f}")

    print(f"{'─'*50}")
    print(f"  {'TP':<8s}: {metrics_dict.get('true_positives', 'N/A')}")
    print(f"  {'TN':<8s}: {metrics_dict.get('true_negatives', 'N/A')}")
    print(f"  {'FP':<8s}: {metrics_dict.get('false_positives', 'N/A')}")
    print(f"  {'FN':<8s}: {metrics_dict.get('false_negatives', 'N/A')}")
    print(f"{'='*50}\n")


def print_comparison_table(results_dict):
    """
    Print comparison table matching TABLE 1 from the paper.

    Args:
        results_dict: {"ModelName": metrics_dict, ...}
    """
    metrics_to_show = ["accuracy", "precision", "recall", "f1_score"]
    header = f"{'Metric':<22s}"
    for model_name in results_dict:
        header += f"| {model_name:<18s}"
    
    print(f"\n{'='*len(header)}")
    print("  PERFORMANCE COMPARISON ON CFTR ΔF508 DATASET")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for metric in metrics_to_show:
        row = f"  {metric:<20s}"
        for model_name, metrics in results_dict.items():
            val = metrics.get(metric, 0)
            row += f"| {val:<18.4f}"
        print(row)

    print(f"{'='*len(header)}\n")


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.uniform(0, 1, 100)

    metrics = compute_all_metrics(y_true, y_proba)
    print_metrics_table(metrics, "Test Metrics")

    # Test deletion-specific
    has_del = np.random.randint(0, 2, 100)
    del_metrics = compute_deletion_specific_metrics(y_true, y_proba, has_del)
    print(f"Deletion recall gain: {del_metrics['deletion_recall_gain']:.4f}")
