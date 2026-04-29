"""
Threshold tuning: sweep decision thresholds on val set, pick best F1, evaluate on test.
Uses the saved test_predictions.npz from full_train.py and re-runs val predictions.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

from configs.config import config
from utils.helpers import set_seed, get_device
from data.encoding import create_dataloaders


def predict_loader(model, loader, device):
    """Return (labels, probs, has_deletion) for a loader."""
    model.eval()
    ys, ps, hds = [], [], []
    with torch.no_grad():
        for batch in loader:
            encoded = batch["encoded"].to(device)
            sgrna = batch["sgrna_seq"]; dna = batch["dna_seq"]
            out = model(encoded, sgrna, dna)
            probs = torch.sigmoid(out["risk_logit"]).cpu().numpy().flatten()
            ys.append(batch["label"].numpy().flatten())
            ps.append(probs)
            hds.append(batch["has_deletion"].numpy().flatten())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(hds)


def metrics_at(y, p, t, hd=None):
    preds = (p > t).astype(int)
    m = {
        "threshold": t,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "mcc": matthews_corrcoef(y, preds),
    }
    if hd is not None and hd.sum() > 0 and (hd == 0).sum() > 0:
        m["recall_deletion"] = recall_score(y[hd == 1], preds[hd == 1], zero_division=0)
        m["recall_no_deletion"] = recall_score(y[hd == 0], preds[hd == 0], zero_division=0)
    cm = confusion_matrix(y, preds)
    if cm.shape == (2, 2):
        m["TN"], m["FP"], m["FN"], m["TP"] = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return m


def main():
    set_seed(config.seed)
    device = get_device()

    df = pd.read_csv(os.path.join(config.data.output_dir, "crispr_dataset.csv"))
    loaders = create_dataloaders(
        df, encoder="null_tensor",
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )

    print("Loading best model...")
    from models.neuro_crispr_kan import NeuroCRISPRKAN
    model = NeuroCRISPRKAN(config).to(device)
    ckpt_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Predicting on val set...")
    y_val, p_val, hd_val = predict_loader(model, loaders["val"], device)
    val_auroc = roc_auc_score(y_val, p_val)
    print(f"  Val AUROC: {val_auroc:.4f}")

    print("Predicting on test set...")
    y_test, p_test, hd_test = predict_loader(model, loaders["test"], device)
    test_auroc = roc_auc_score(y_test, p_test)
    print(f"  Test AUROC: {test_auroc:.4f}")

    # Sweep thresholds 0.05–0.95
    thresholds = np.arange(0.05, 0.96, 0.01)
    val_results = [metrics_at(y_val, p_val, t, hd_val) for t in thresholds]

    # Best by F1 on val
    best_f1 = max(val_results, key=lambda r: r["f1"])
    # Best by MCC on val
    best_mcc = max(val_results, key=lambda r: r["mcc"])

    print("\n" + "="*70)
    print("  THRESHOLD SWEEP (best on VAL)")
    print("="*70)
    print(f"  Best F1 threshold:  t={best_f1['threshold']:.2f}  val F1={best_f1['f1']:.4f}  Prec={best_f1['precision']:.4f}  Rec={best_f1['recall']:.4f}")
    print(f"  Best MCC threshold: t={best_mcc['threshold']:.2f}  val MCC={best_mcc['mcc']:.4f}")

    # Apply best F1 threshold to TEST
    print("\n" + "="*70)
    print(f"  TEST RESULTS at threshold = {best_f1['threshold']:.2f} (best F1 on val)")
    print("="*70)
    test_at_best = metrics_at(y_test, p_test, best_f1["threshold"], hd_test)
    test_at_best["auroc"] = test_auroc
    for k, v in test_at_best.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    # Compare to default 0.5
    print("\n" + "="*70)
    print("  TEST RESULTS at threshold = 0.50 (default, for comparison)")
    print("="*70)
    test_at_default = metrics_at(y_test, p_test, 0.50, hd_test)
    test_at_default["auroc"] = test_auroc
    for k, v in test_at_default.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    # Save sweep
    out = {
        "val_auroc": val_auroc,
        "test_auroc": test_auroc,
        "best_threshold_f1": best_f1["threshold"],
        "best_threshold_mcc": best_mcc["threshold"],
        "test_at_best_threshold": test_at_best,
        "test_at_default": test_at_default,
        "sweep": val_results,
    }
    out_path = os.path.join(config.training.checkpoint_dir, "threshold_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved sweep to {out_path}")


if __name__ == "__main__":
    main()
