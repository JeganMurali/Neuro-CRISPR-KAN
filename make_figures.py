"""
Generate all defense/paper figures from saved training artifacts.
Outputs to ./figures/.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
})

OUT = "./figures"
os.makedirs(OUT, exist_ok=True)

CKPT = "./checkpoints"
NT_HIST = json.load(open(f"{CKPT}/history.json"))
ZP_HIST = json.load(open(f"{CKPT}/ablation_zeropad/history.json"))
SWEEP = json.load(open(f"{CKPT}/threshold_sweep.json"))
NT_PRED = np.load(f"{CKPT}/test_predictions.npz")
ZP_PRED = np.load(f"{CKPT}/ablation_zeropad/test_predictions.npz")


# ---- Fig 1: Training curves (loss + AUROC) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
nt_ep = [h["epoch"] for h in NT_HIST]; zp_ep = [h["epoch"] for h in ZP_HIST]
ax = axes[0]
ax.plot(nt_ep, [h["train_loss"] for h in NT_HIST], "-", color="C0", label="Null Tensor — train", alpha=0.6)
ax.plot(nt_ep, [h["val_loss"]   for h in NT_HIST], "-", color="C0", label="Null Tensor — val",   linewidth=2)
ax.plot(zp_ep, [h["train_loss"] for h in ZP_HIST], "--", color="C3", label="Zero-Pad — train",   alpha=0.6)
ax.plot(zp_ep, [h["val_loss"]   for h in ZP_HIST], "--", color="C3", label="Zero-Pad — val",     linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Compound loss"); ax.set_title("Training & Validation Loss")
ax.legend(loc="best")

ax = axes[1]
ax.plot(nt_ep, [h["val_auroc"] for h in NT_HIST], "-", color="C0", linewidth=2.2, label="Null Tensor")
ax.plot(zp_ep, [h["val_auroc"] for h in ZP_HIST], "--", color="C3", linewidth=2.2, label="Zero-Pad")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="random")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUROC"); ax.set_title("Validation AUROC over Training")
ax.set_ylim(0.45, 0.95); ax.legend(loc="best")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_training_curves.png", bbox_inches="tight"); plt.close()
print(f"[1] {OUT}/fig1_training_curves.png")


# ---- Fig 2: ROC curves ----
y_nt, p_nt = NT_PRED["y"], NT_PRED["p"]
y_zp, p_zp = ZP_PRED["y"], ZP_PRED["p"]
fpr_nt, tpr_nt, _ = roc_curve(y_nt, p_nt); auc_nt = auc(fpr_nt, tpr_nt)
fpr_zp, tpr_zp, _ = roc_curve(y_zp, p_zp); auc_zp = auc(fpr_zp, tpr_zp)
fig, ax = plt.subplots(figsize=(6.2, 5.5))
ax.plot(fpr_nt, tpr_nt, color="C0", linewidth=2.4, label=f"Null Tensor (AUC={auc_nt:.3f})")
ax.plot(fpr_zp, tpr_zp, color="C3", linewidth=2.4, linestyle="--", label=f"Zero-Pad (AUC={auc_zp:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.7, label="random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Test-set ROC: Null Tensor vs Zero-Padding")
ax.legend(loc="lower right"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_roc_curves.png", bbox_inches="tight"); plt.close()
print(f"[2] {OUT}/fig2_roc_curves.png")


# ---- Fig 3: Confusion matrices at t=0.5 and t=0.7 (Null Tensor) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, t in zip(axes, [0.5, 0.7]):
    preds = (p_nt > t).astype(int)
    cm = confusion_matrix(y_nt, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Safe", "Dangerous"], yticklabels=["Safe", "Dangerous"], ax=ax,
                annot_kws={"size": 14})
    acc = accuracy_score(y_nt, preds); rec = recall_score(y_nt, preds); prec = precision_score(y_nt, preds, zero_division=0)
    ax.set_title(f"Threshold = {t:.1f}\nAcc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Null Tensor: Confusion matrices at two thresholds", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_confusion_matrices.png", bbox_inches="tight"); plt.close()
print(f"[3] {OUT}/fig3_confusion_matrices.png")


# ---- Fig 4: Deletion-stratified recall comparison ----
def stratified_recall(y, p, hd, t=0.5):
    preds = (p > t).astype(int)
    full = recall_score(y, preds, zero_division=0)
    rd = recall_score(y[hd==1], preds[hd==1], zero_division=0) if (hd==1).sum() else float("nan")
    rn = recall_score(y[hd==0], preds[hd==0], zero_division=0) if (hd==0).sum() else float("nan")
    return full, rd, rn

hd_nt = NT_PRED["has_deletion"]; hd_zp = ZP_PRED["has_deletion"]
nt_full, nt_del, nt_nodel = stratified_recall(y_nt, p_nt, hd_nt)
zp_full, zp_del, zp_nodel = stratified_recall(y_zp, p_zp, hd_zp)

categories = ["Overall", "ΔF508 deletion", "No deletion"]
nt_vals = [nt_full, nt_del, nt_nodel]
zp_vals = [zp_full, zp_del, zp_nodel]
x = np.arange(len(categories)); w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 5))
b1 = ax.bar(x - w/2, nt_vals, w, label="Null Tensor", color="C0", edgecolor="black")
b2 = ax.bar(x + w/2, zp_vals, w, label="Zero-Pad", color="C3", edgecolor="black")
for bar, v in list(zip(b1, nt_vals)) + list(zip(b2, zp_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
# Delta annotations
for i, (a, b) in enumerate(zip(nt_vals, zp_vals)):
    delta = a - b
    ax.text(x[i], 0.05, f"Δ +{delta:.3f}", ha="center", fontsize=10, color="darkgreen", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_ylabel("Recall (Sensitivity)"); ax.set_title("Recall at threshold 0.5: deletion-stratified comparison")
ax.set_ylim(0, 1.0); ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_deletion_stratified_recall.png", bbox_inches="tight"); plt.close()
print(f"[4] {OUT}/fig4_deletion_stratified_recall.png")


# ---- Fig 5: Threshold sweep ----
sweep = SWEEP["sweep"]
ts = [s["threshold"] for s in sweep]
prec = [s["precision"] for s in sweep]
rec  = [s["recall"]    for s in sweep]
f1   = [s["f1"]        for s in sweep]
mcc  = [s["mcc"]       for s in sweep]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ts, prec, label="Precision", linewidth=2)
ax.plot(ts, rec,  label="Recall",    linewidth=2)
ax.plot(ts, f1,   label="F1",        linewidth=2)
ax.plot(ts, mcc,  label="MCC",       linewidth=2, linestyle=":")
ax.axvline(SWEEP["best_threshold_f1"], color="black", linestyle="--", alpha=0.6,
           label=f"best F1 t={SWEEP['best_threshold_f1']:.2f}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold sweep on validation set (Null Tensor)")
ax.legend(loc="best")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_threshold_sweep.png", bbox_inches="tight"); plt.close()
print(f"[5] {OUT}/fig5_threshold_sweep.png")


# ---- Fig 6: Test metrics summary (vs paper baselines) ----
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUROC"]
deepcrispr = [0.87, 0.84, 0.81, 0.82, np.nan]
crisprnet  = [0.91, 0.89, 0.85, 0.87, np.nan]
zeropad_t  = [0.711, 0.504, 0.792, 0.616, 0.817]
nulltensor_t = [0.731, 0.524, 0.863, 0.652, 0.873]
paper_target = [0.94, 0.93, 0.89, 0.91, np.nan]

x = np.arange(len(metrics)); w = 0.16
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - 2*w, deepcrispr,    w, label="DeepCRISPR (paper)", color="#999999")
ax.bar(x - w,   crisprnet,     w, label="CRISPR-Net (paper)", color="#666666")
ax.bar(x,       zeropad_t,     w, label="Ours (Zero-Pad)",   color="C3")
ax.bar(x + w,   nulltensor_t,  w, label="Ours (Null Tensor)", color="C0")
ax.bar(x + 2*w, paper_target,  w, label="Paper target", color="C2", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("Score (test set, t=0.5)"); ax.set_ylim(0, 1.0)
ax.set_title("Comparison: baselines (paper-reported) vs our trained models")
ax.legend(loc="lower right", ncol=2)
for i, vals in enumerate(zip(deepcrispr, crisprnet, zeropad_t, nulltensor_t, paper_target)):
    for j, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(x[i] + (j-2)*w, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_baseline_comparison.png", bbox_inches="tight"); plt.close()
print(f"[6] {OUT}/fig6_baseline_comparison.png")


# ---- Fig 7: Risk-score distribution by class ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (y, p, title) in zip(axes,
                              [(y_nt, p_nt, "Null Tensor"), (y_zp, p_zp, "Zero-Pad")]):
    ax.hist(p[y == 0], bins=40, alpha=0.6, label="Safe (label=0)", color="C2", edgecolor="black")
    ax.hist(p[y == 1], bins=40, alpha=0.6, label="Dangerous (label=1)", color="C3", edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Predicted risk probability"); ax.set_ylabel("Count")
    ax.set_title(f"{title}: predicted risk distribution"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_risk_distribution.png", bbox_inches="tight"); plt.close()
print(f"[7] {OUT}/fig7_risk_distribution.png")

print(f"\nAll figures saved to {OUT}/")
print("Files:")
for f in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{f}")
