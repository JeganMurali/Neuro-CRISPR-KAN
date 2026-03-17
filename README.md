# Neuro-CRISPR-KAN

**A Hybrid CNN-Transformer Architecture for Off-Target Prediction in Cystic Fibrosis**

> IEEE ICAUC 2026 | Paper ID: ICAUC-500

---

## Architecture Overview

```
sgRNA-DNA Pair
      │
      ▼
┌─────────────────┐
│  Null Tensor     │  (Deletion-aware encoding, NOT zero-padding)
│  Encoding        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ 1D-CNN │ │  DNABERT-2   │
│ Stream │ │  (LoRA)      │
└───┬────┘ └──────┬───────┘
    │              │
    └──────┬───────┘
           ▼
    ┌─────────────┐
    │   Feature    │
    │   Fusion     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  KAN Core   │  (B-spline learnable activations)
    │  Decision   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Risk Score  │──► RAG + LLM ──► Safety Audit Summary
    └─────────────┘
```

## Project Structure

```
neuro_crispr_kan/
├── configs/
│   └── config.py              # All hyperparameters & paths
├── data/
│   ├── data_generation.py     # Synthetic sgRNA-DNA dataset (10K rows)
│   └── encoding.py            # Null Tensor + zero-padding encoders
├── models/
│   ├── cnn_stream.py          # 1D-CNN for local motif extraction
│   ├── transformer_stream.py  # DNABERT-2 with LoRA adapter
│   ├── kan_layer.py           # Custom KAN with B-spline edges
│   ├── fusion.py              # Feature fusion module
│   └── neuro_crispr_kan.py    # Full assembled model
├── training/
│   ├── losses.py              # Focal loss + spline regularization
│   ├── train.py               # Training loop with cosine annealing
│   └── optimizer.py           # Adam + scheduler setup
├── evaluation/
│   ├── metrics.py             # Accuracy, Precision, Recall, F1, Spearman
│   ├── evaluate.py            # Full evaluation pipeline
│   ├── ablation.py            # Null Tensor vs Zero-Padding ablation
│   └── visualize.py           # Attention heatmaps & metric plots
├── rag/
│   └── rag_llm.py             # RAG module + LLM safety audit generation
├── ui/
│   └── app.py                 # Streamlit dashboard
├── utils/
│   └── helpers.py             # Seed setting, device utils, logging
├── notebooks/
│   └── colab_runner.ipynb     # Single notebook to run everything on Colab
├── requirements.txt
└── README.md
```

## Setup (Google Colab)

```bash
!pip install torch transformers peft biopython chromadb streamlit plotly scikit-learn scipy
!git clone <your-repo-url>
%cd neuro_crispr_kan
```

## 2-Week Implementation Timeline

| Week | Days | Modules | Goal |
|------|------|---------|------|
| 1 | 1-2 | `configs`, `data/`, `utils/` | Dataset ready, encoding verified |
| 1 | 3-4 | `models/cnn_stream`, `models/kan_layer` | CNN + KAN working |
| 1 | 5-7 | `models/transformer_stream`, `models/fusion`, `models/neuro_crispr_kan` | Full model forward pass |
| 2 | 8-10 | `training/` | Model trained, checkpoints saved |
| 2 | 11-12 | `evaluation/` | All metrics computed, ablation done |
| 2 | 13-14 | `rag/`, `ui/` | Safety audits + Streamlit demo |

## Key Metrics (from paper)

| Metric | DeepCRISPR | CRISPR-Net | **Neuro-CRISPR-KAN** |
|--------|-----------|------------|---------------------|
| Accuracy | 0.87 | 0.91 | **0.94** |
| Precision | 0.84 | 0.89 | **0.93** |
| Recall | 0.81 | 0.85 | **0.89** |
| F1-Score | 0.82 | 0.87 | **0.91** |
| Spearman ρ | 0.79 | 0.84 | **0.88** |
