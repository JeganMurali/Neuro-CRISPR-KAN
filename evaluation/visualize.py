"""
Visualization Module
====================
Generates all plots and visualizations for the project:

1. Attention heatmaps (from Transformer stream)
2. Training curves (loss, accuracy over epochs)
3. ROC & Precision-Recall curves
4. Comparison bar charts (Table 1 visual)
5. B-spline activation visualization
6. Confusion matrix heatmap
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab
import seaborn as sns
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import get_roc_curve_data, get_pr_curve_data


SAVE_DIR = "./plots"


def _ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def plot_training_curves(history, save=True):
    """Plot training and validation loss/accuracy curves."""
    _ensure_dir()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation accuracy
    axes[1].plot(epochs, val_acc, "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    lrs = [h["lr"] for h in history]
    axes[2].plot(epochs, lrs, "purple", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Cosine Annealing LR Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5, save=True):
    """Plot confusion matrix heatmap."""
    _ensure_dir()
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Safe (0)", "Off-target (1)"],
        yticklabels=["Safe (0)", "Off-target (1)"],
        ax=ax, annot_kws={"size": 14}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Neuro-CRISPR-KAN Confusion Matrix", fontsize=14)

    if save:
        plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save=True):
    """Plot ROC curve with AUROC."""
    _ensure_dir()
    roc_data = get_roc_curve_data(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_true, y_pred_proba)

    ax.plot(roc_data["fpr"], roc_data["tpr"], "b-", linewidth=2,
            label=f"Neuro-CRISPR-KAN (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison_bars(results_dict, save=True):
    """
    Bar chart comparing models — visual version of Table 1.

    Args:
        results_dict: {"ModelName": {"accuracy": ..., "precision": ..., ...}}
    """
    _ensure_dir()
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = list(results_dict.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#1E88E5"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [results_dict[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model,
                      color=colors[i % len(colors)], edgecolor="white")
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison on CFTR ΔF508 Dataset")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    if save:
        plt.savefig(os.path.join(SAVE_DIR, "comparison_bars.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_heatmap(attention_weights, tokens=None, layer=-1, head=0, save=True):
    """
    Plot attention heatmap from Transformer stream.

    Args:
        attention_weights: Tuple of attention tensors from DNABERT-2
        tokens: List of token strings for axis labels
        layer: Which attention layer to visualize (-1 = last)
        head: Which attention head
    """
    _ensure_dir()
    # Get specific layer's attention
    attn = attention_weights[layer][0, head].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn, cmap="YlOrRd", ax=ax,
        xticklabels=tokens if tokens else False,
        yticklabels=tokens if tokens else False,
    )
    ax.set_title(f"Self-Attention Heatmap (Layer {layer}, Head {head})")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    if save:
        plt.savefig(os.path.join(SAVE_DIR, "attention_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation_comparison(null_tensor_metrics, zero_pad_metrics, save=True):
    """
    Side-by-side comparison of Null Tensor vs Zero-Padding.
    Highlights the recall gain for deletion-specific samples.
    """
    _ensure_dir()
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall\n(Key Metric)", "F1-Score"]

    nt_vals = [null_tensor_metrics.get(m, 0) for m in metrics]
    zp_vals = [zero_pad_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, nt_vals, width, label="Null Tensor (Proposed)",
                   color="#1E88E5", edgecolor="white")
    bars2 = ax.bar(x + width/2, zp_vals, width, label="Zero-Padding (Baseline)",
                   color="#FF6B6B", edgecolor="white")

    for bar, val in zip(bars1, nt_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, zp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("Ablation: Null Tensor vs Zero-Padding Encoding")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    if save:
        plt.savefig(os.path.join(SAVE_DIR, "ablation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Generate dummy plots for testing
    np.random.seed(42)

    # Dummy training history
    history = [
        {"epoch": i, "train_loss": 0.5 * np.exp(-0.05*i) + np.random.normal(0, 0.02),
         "val_loss": 0.55 * np.exp(-0.04*i) + np.random.normal(0, 0.03),
         "val_accuracy": 0.6 + 0.3 * (1 - np.exp(-0.08*i)),
         "lr": 1e-4 * (1 + np.cos(np.pi * i / 50)) / 2}
        for i in range(50)
    ]
    plot_training_curves(history)
    print("Training curves saved.")

    # Dummy comparison
    comparison = {
        "DeepCRISPR": {"accuracy": 0.87, "precision": 0.84, "recall": 0.81, "f1_score": 0.82},
        "CRISPR-Net": {"accuracy": 0.91, "precision": 0.89, "recall": 0.85, "f1_score": 0.87},
        "Neuro-CRISPR-KAN": {"accuracy": 0.94, "precision": 0.93, "recall": 0.89, "f1_score": 0.91},
    }
    plot_comparison_bars(comparison)
    print("Comparison chart saved.")
