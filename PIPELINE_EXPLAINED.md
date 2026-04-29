# How Your Project Works (End-to-End)

> A plain-language walkthrough of the Neuro-CRISPR-KAN pipeline using **real data from your generated dataset**.

---

## 🎯 The Problem You're Solving

CRISPR-Cas9 is a "molecular scissors" that cuts DNA at a specific location to fix mutations (like ΔF508 in Cystic Fibrosis). But sometimes Cas9 cuts the **wrong** location — that's an "off-target" cut. Off-targets can cause cancer or new diseases.

**Your model answers:** *"Given this guide RNA + this DNA target, will Cas9 cut it (DANGEROUS=1) or not (SAFE=0)?"*

---

## 1️⃣ THE DATASET (10,000 synthetic samples)

Generated to `./data/generated/crispr_dataset.csv`. Each row has 11 columns. Look at **Sample 0**:

```
sgRNA            : GATAGGCATAAGAAGGAGCATGG    ← 23 letters (the "guide", what we WANT to cut)
DNA target       : GATAGGTATAAGGAGCTTGG       ← 20 letters (the actual DNA, 3 missing = ΔF508)
has_deletion     : 1                          ← Yes, ΔF508 is present
num_mismatches   : 2                          ← 2 letters differ between sgRNA and DNA
mismatch_pos     : [6, 16]                    ← positions where they differ
seed_mismatches  : 1                          ← mismatches in the critical "seed region" (positions 1-12)
pam_intact       : 1                          ← The "GG" at end is preserved (Cas9 needs this to cut)
chromatin_score  : 0.3537                     ← How "open" the DNA is (0 = closed, 1 = wide open)
off_target_label : 1                          ← 🚨 DANGEROUS — Cas9 will cut here
efficiency_score : 0.6860                     ← How efficiently Cas9 cleaves (regression target)
```

### Why each field matters biologically

- **PAM site (NGG)**: last 3 letters. Cas9 *requires* this to bind. No PAM → no cut.
- **Seed region (positions 1-12 from PAM)**: mismatches here are *much* worse than mismatches further away.
- **ΔF508 deletion**: 3 letters (CTT) deleted at position 12 in the CFTR gene. This is the actual mutation that causes Cystic Fibrosis.
- **Chromatin score**: tightly packed DNA is hard for Cas9 to reach; loose DNA is easy.

### Dataset stats (matches paper)

| | |
|---|---|
| Total samples | 10,000 |
| Dangerous (label=1) | 2,637 (26.4%) — **rare event problem** |
| With ΔF508 deletion | 4,042 (40.4%) |
| Average mismatches | 1.88 |
| Splits | 7,000 train / 1,500 val / 1,500 test |

---

## 2️⃣ ENCODING — Turning Letters into Numbers

Neural nets can't read "ATCG". We turn each letter into a number vector.

### Standard one-hot (4 channels) — what BASELINES use

```
A → [1,0,0,0]
T → [0,1,0,0]
G → [0,0,1,0]
C → [0,0,0,1]
```

### Your "Null Tensor" (5 channels) — the paper's KEY INNOVATION

```
A   → [1,0,0,0,0]
T   → [0,1,0,0,0]
G   → [0,0,1,0,0]
C   → [0,0,0,1,0]
GAP → [0,0,0,0,1]    ← 5th channel explicitly marks deletion
```

### Sample 0's DNA encoding — side-by-side

The DNA `GATAGGTATAAGGAGCTTGG` (20bp, has ΔF508 deletion) gets placed into a 23-position tensor:

```
position 11:  G   →  NULL  [0,0,1,0,0]  |  ZEROPAD [0,0,1,0]    SAME
position 12:  GAP →  NULL  [0,0,0,0,1]  |  ZEROPAD [0,0,1,0]    ← Different! Null says "GAP", zeropad says "G"
position 13:  GAP →  NULL  [0,0,0,0,1]  |  ZEROPAD [1,0,0,0]    ← Different!
position 14:  GAP →  NULL  [0,0,0,0,1]  |  ZEROPAD [0,0,1,0]    ← Different!
position 15:  G   →  NULL  [0,0,1,0,0]  |  ZEROPAD [0,0,0,1]    ← Wrong letter!
...
position 20-22: T,G,G → NULL keeps them | ZEROPAD says [0,0,0,0]  ← Lost!
```

**This is THE problem:** Zero-padding shifts every letter after the deletion **left by 3 positions** and dumps zeros at the end. The model sees `G` at position 15 instead of `G` at position 18 — totally wrong! The PAM site (which should always be at positions 20-22) ends up at positions 17-19 → model can't find it consistently.

**Null Tensor fixes this:** every letter stays at its correct position. The 3 missing letters are explicitly marked as "GAP" via the 5th channel. The model learns: *"channel 5 firing = there's a deletion here"*.

This single idea is why the paper claims a ~10% recall gain on deletion-specific off-targets.

---

## 3️⃣ THE MODEL — Two Eyes Looking at the Same DNA

The encoded sequence (2 strands × 23 positions × 5 channels) goes into **two parallel networks** that look at it differently:

```
                  Encoded sgRNA-DNA pair (2, 23, 5)
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       ┌────────────┐              ┌──────────────┐
       │ CNN STREAM │              │ TRANSFORMER  │
       │            │              │   STREAM     │
       │ "Magnifying│              │              │
       │  glass"    │              │  "Reading    │
       │  Looks at  │              │   the whole  │
       │  3-7 letter│              │   sequence"  │
       │  patterns  │              │              │
       └─────┬──────┘              └──────┬───────┘
             │ (B, 128)                   │ (B, 128)
             └───────────┬────────────────┘
                         ▼
                ┌─────────────────┐
                │ FUSION (gate)   │  ← learns: "trust CNN more for deletions,
                │                 │     trust Transformer more for context"
                └────────┬────────┘
                         │ (B, 256)
                         ▼
                ┌─────────────────┐
                │ KAN DECISION    │  ← learnable spline activations
                │  CORE           │     (better than fixed ReLU/sigmoid)
                └────────┬────────┘
                         │
                         ▼
                  risk_prob ∈ [0,1]
                         │
                         ▼
                ┌─────────────────┐
                │ RAG + LLM       │  ← turns "0.86" into:
                │  Safety Audit   │     "HIGH risk: 1 mismatch in seed region,
                └─────────────────┘      PAM intact, recommend redesign"
```

### CNN Stream (`models/cnn_stream.py`)

- Like a **magnifying glass** scanning across the sequence.
- 3 parallel convolutions with kernel sizes **3, 5, 7** = sees patterns of 3, 5, 7 nucleotides at once.
  - kernel 3 → catches PAM (NGG, 3 letters)
  - kernel 5 → catches PAM + context
  - kernel 7 → catches longer seed motifs
- Each branch uses **residual blocks** (skip connections) so gradients flow well.
- Output: 128-dim feature vector summarizing **local structure**.

### Transformer Stream (`models/transformer_stream.py`)

- Uses **DNABERT-2** — a pre-trained model that has already learned DNA "language" from billions of genome bases.
- Reads the *entire* sgRNA+DNA together and understands *long-range relationships*.
- We don't retrain all 117M parameters (would overfit) — we use **LoRA**: freeze the original, add tiny adapter layers (only **294,912 new trainable params** = 0.25%).
- Output: another 128-dim feature vector summarizing **global semantics**.

**What is LoRA?**
> LoRA = Low-Rank Adaptation. Instead of updating the full 768×768 attention matrix `W`, we add a tiny `A·B` correction (where `A` is 768×8 and `B` is 8×768). The original `W` stays frozen. Only the small matrices train. 400× fewer parameters, almost the same flexibility.

### Fusion (`models/fusion.py`)

- Combines CNN's local view + Transformer's global view.
- A **gate** learns weights (e.g., `[0.6 CNN, 0.4 Transformer]`) per sample.
- The gate is itself a small linear layer that adapts to each input.
- Output: 256-dim fused representation.

### KAN Decision Core (`models/kan_layer.py`) — the 2nd KEY INNOVATION

- Standard MLP uses **fixed activations** (ReLU just does `max(0,x)`). Limited.
- KAN replaces them with **B-splines** = smooth curves with 10 learnable control points per edge.
- Result: each connection in the network learns its *own custom activation shape*.
- Better at modeling **sharp, rare-event boundaries** — exactly what off-targets are (rare!).
- Three layers: `256 → 128 → 64 → 1`.
- Outputs a single number → sigmoid → `risk_prob ∈ [0, 1]`.

**Why splines beat ReLU here:**
> Off-target risk is "spiky" — small changes (1 mismatch in the seed) cause big jumps in risk. ReLU networks have a "spectral bias" — they prefer smooth, gentle curves and can't model sharp jumps well. KAN's splines bend exactly where the data needs them to.

### RAG + LLM (`rag/rag_llm.py`)

- Takes the risk score (e.g. 0.86) and the sample features.
- Retrieves relevant biology snippets from a small knowledge base (PAM mechanism, seed importance, etc.) using sentence embeddings.
- A small LLM (flan-T5) writes a human-readable safety report grounded in those retrieved snippets.
- This is the "interpretability" piece — turns numbers into explanations a biologist can act on.

---

## 4️⃣ TRAINING — How the Model Learns

```
For each of 50 epochs:
    For each batch of 64 samples:
        1. Forward pass → get predicted risk
        2. Compute loss = how wrong we were
        3. Backward pass → compute gradients
        4. Adam optimizer → tweak ~1.1M trainable params
    Validate on val set → save if improved
```

### The loss = `0.7 × Focal Loss + 0.25 × Spline L1`

**Focal Loss** (handles imbalance):
- Standard loss treats all errors equally → with 74% safe samples, model just predicts "safe" always for free 74% accuracy.
- Focal loss says: *"if you're already correct and confident, I won't waste training time on you. Focus on hard, misclassified samples."*
- Formula: `-α · (1 - p_t)^γ · log(p_t)` with `γ=2.0`.
- Critical because off-targets are rare and we **really** don't want to miss them.

**Spline L1**: keeps KAN curves smooth (prevents wild overfitting where splines wiggle too much).

### Schedule

- **Adam** optimizer with **cosine annealing** (LR starts at 1e-4, slowly drops to 1e-7 over 50 epochs).
- **Early stopping**: if validation loss doesn't improve for 7 epochs → stop.
- **Gradient clipping** at norm 1.0: keeps gradients bounded so training is stable.

### Different learning rates for different parts

- **DNABERT-2 LoRA** (pre-trained) → **1e-5** (tiny tweaks, don't break what it already knows)
- **CNN + KAN** (from scratch) → **1e-4** (faster learning)

---

## 5️⃣ EVALUATION METRICS

After training, we measure on the test set (1,500 unseen samples):

| Metric | Meaning | Why it matters |
|---|---|---|
| **Accuracy** | % predictions correct | Easy to fool with imbalanced data |
| **Precision** | Of flagged dangerous, how many really are? | High = few false alarms |
| **Recall** (Sensitivity) | Of actually dangerous, how many caught? | **MOST IMPORTANT** — don't miss real threats! |
| **F1-Score** | Balance of precision + recall | Single number for overall quality |
| **Specificity** | Of safe, how many correctly called safe? | Avoid over-warning |
| **MCC** | Balanced metric for imbalanced data | Robust correctness |
| **AUROC** | Overall ranking quality | Independent of threshold |
| **Spearman ρ** | Predicted vs true cleavage rank | Quantitative agreement |

### Paper's claimed results (Table 1)

| Metric | DeepCRISPR | CRISPR-Net | **Neuro-CRISPR-KAN** |
|---|---|---|---|
| Accuracy | 0.87 | 0.91 | **0.94** |
| Precision | 0.84 | 0.89 | **0.93** |
| Recall | 0.81 | 0.85 | **0.89** |
| F1 | 0.82 | 0.87 | **0.91** |
| Spearman ρ | 0.79 | 0.84 | **0.88** |

---

## 6️⃣ WHAT'S BEEN VERIFIED ON YOUR A100

### Environment & dependencies
| Step | Status |
|---|---|
| Conda env `neuro_crispr` (Python 3.11, torch 2.6 + CUDA 12.4) | ✅ |
| All requirements installed (torch, transformers 4.29.2, peft 0.7.1, accelerate 0.27.2, einops, etc.) | ✅ |
| DNABERT-2 downloaded (117M params) | ✅ |
| Patched DNABERT-2 `bert_layers.py` to disable broken Triton flash-attention (incompatible with Triton 3.x) | ✅ |
| Removed unsupported `attn_implementation="eager"` kwarg in `models/transformer_stream.py:71` | ✅ |

### Data & encoding
| Step | Status |
|---|---|
| Dataset generated (10K rows, matches paper stats) | ✅ at `./data/generated/crispr_dataset.csv` |
| 26.4% positives, 40.4% with ΔF508 deletion | ✅ |
| Null Tensor encoding produces (2, 23, 5) with explicit GAP channel | ✅ |
| GAP channel fires at exactly positions 12, 13, 14 for every deletion sample | ✅ |
| DataLoader passes `has_deletion` to encoder correctly (verified across batch of 64, zero inconsistencies) | ✅ |

### Model components (all forward passes verified on GPU)
| Step | Status |
|---|---|
| CNN stream: (B, 2, 23, 5) → (B, 128) | ✅ |
| DNABERT-2 + LoRA: strings → (B, 128) — 294,912 trainable params (0.25%) | ✅ |
| Fusion: two (B, 128) → (B, 256) | ✅ |
| KAN core: (B, 256) → risk score (B, 1) | ✅ |
| Full model: 118M total params, 1.1M trainable, peak GPU 0.5 GB at batch=4 | ✅ |

### Training pipeline (5-epoch smoke train, constant LR)
| Metric | Value |
|---|---|
| End-to-end run completed without errors | ✅ |
| Total time on A100 | **1.0 min** (~11s/epoch) |
| GPU throughput | 11.5 it/s training, 21 it/s val |
| Focal loss decreased | 0.111 → 0.081 (–26%) ✅ |
| Spline-reg loss decreased | 0.078 → 0.072 (–8%) ✅ |
| Val accuracy (unstable, threshold-bound) | 0.62 → 0.48 ⚠ |
| Test AUROC after 5 epochs | 0.507 (random) ⚠ |
| Test recall after 5 epochs | 0.516 ⚠ |

**Interpretation:** plumbing is healthy and gradients flow, but 5 epochs is far too few for the 118M-parameter model to learn discrimination — it's still learning the prior. Real training run (50 epochs) is required to evaluate paper claims.

### What was learned about the training loop
- The repo's default `CosineAnnealingLR` with `T_max=cfg.epochs` collapses LR to ~0 within `cfg.epochs` steps. For short runs this kills learning before it starts. Either run the full 50 epochs as the paper specifies, or override the scheduler for shorter runs (see `smoke_train.py`).
- Optimizer has 4 param groups: DNABERT-2 LoRA at lr=1e-5 (group 0), CNN + KAN + Fusion all at lr=1e-4 (groups 1–3). The 10× lower LR on LoRA is intentional and correct.

---

## 7️⃣ FILE-BY-FILE MAP

```
Neuro-CRISPR-KAN/
├── configs/config.py              # All hyperparameters in one place
├── data/
│   ├── data_generation.py         # Generate the 10K synthetic dataset
│   └── encoding.py                # Null Tensor (5-ch) and Zero-Pad (4-ch) encoders
├── models/
│   ├── cnn_stream.py              # 1D-CNN with 3 kernel branches + residuals
│   ├── transformer_stream.py      # DNABERT-2 + LoRA wrapper
│   ├── fusion.py                  # Gated fusion module
│   ├── kan_layer.py               # B-spline KAN layers + decision core
│   └── neuro_crispr_kan.py        # Full assembled model
├── training/
│   ├── losses.py                  # Focal loss + spline L1
│   ├── optimizer.py               # Adam + cosine annealing + param groups
│   └── train.py                   # Training loop with early stopping
├── evaluation/
│   ├── metrics.py                 # All metrics (Accuracy, F1, MCC, AUROC, …)
│   ├── evaluate.py                # Run model on test set, print report
│   ├── ablation.py                # Null Tensor vs Zero-Pad comparison
│   └── visualize.py               # Plots: confusion matrix, ROC, attention heatmaps
├── rag/rag_llm.py                 # RAG + flan-T5 safety audit
├── ui/app.py                      # Streamlit dashboard
├── utils/helpers.py               # Seeds, device, logging, checkpoints
├── run_training.py                # Single-entry script: data → model → train → eval
├── smoke_train.py                 # 5-epoch smoke test with inline loop, constant LR
└── PIPELINE_EXPLAINED.md          # ← this file
```

---

## 8️⃣ DATALOADER VERIFICATION (✅ PASSED)

Confirmed end-to-end on the real dataset:

- `CRISPRDataset.__getitem__` correctly passes `has_deletion=bool(row["has_deletion"])` to `encode_pair()` (`data/encoding.py:197`).
- Across a sampled batch of 64: every sample with `has_deletion=True` had the GAP channel firing at exactly positions 12, 13, 14. Every non-deletion sample had zero GAP positions. Zero inconsistencies.
- This means the paper's central claim — that the model sees an **explicit structural deletion signal** rather than zero-padded noise — is wired correctly into training.

---

## 9️⃣ HOW TO RUN

Activate the env first:

```bash
source /home/connect/miniconda3/etc/profile.d/conda.sh
conda activate neuro_crispr
cd /home/connect/Neuro_Crispr/Neuro-CRISPR-KAN
```

| Command | Purpose | Time |
|---|---|---|
| `python smoke_train.py` | 5-epoch smoke test (constant LR) | ~1 min |
| `python run_training.py` | Full 50-epoch training (cosine scheduler) | ~10 min |
| `python -m evaluation.evaluate` | Run trained checkpoint on test set | ~10 s |
| `python -m evaluation.ablation` | Null Tensor vs Zero-Pad comparison | ~20 min (trains both) |
| `streamlit run ui/app.py` | Demo dashboard (currently uses mock predictions) | interactive |

Checkpoints: `./checkpoints/best_model.pt` and `./checkpoints/final_model.pt`
Dataset: `./data/generated/crispr_dataset.csv`
Logs: `./logs/training.log`

---

## 🎓 What You Can Do Next

| | Task | Time | Why |
|---|---|---|---|
| ✅ | Verify dataloader passes `has_deletion` | done | Confirms paper's central claim wired correctly |
| ✅ | 5-epoch smoke train | done | Confirms pipeline works end-to-end |
| ⏭ | **Full 50-epoch training** | ~10 min | Reproduces real model. Adds per-epoch AUROC logging recommended. |
| ⏭ | Ablation: Null Tensor vs Zero-Pad | ~20 min | The paper's central experimental claim (~10% recall gain) |
| ⏭ | Reproduce Table 1 metrics | ~1 min after training | Compares against DeepCRISPR, CRISPR-Net baselines |
| ⏭ | Wire trained model into Streamlit UI | ~30 min | Currently the UI uses a mock formula, not the real model |
| ⏭ | Hook RAG safety audit pipeline | ~30 min | Translates risk → human-readable report |
