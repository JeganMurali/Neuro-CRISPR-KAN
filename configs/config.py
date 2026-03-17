"""
Neuro-CRISPR-KAN Configuration
==============================
All hyperparameters, paths, and constants in one place.
Modify this file to tune the entire pipeline.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset generation & encoding parameters."""
    num_samples: int = 10_000
    sequence_length: int = 23          # Standard sgRNA length (bp)
    deletion_position: int = 12        # ΔF508 deletion center index
    deletion_length: int = 3           # CTT deletion (3 nucleotides)
    null_tensor_dim: int = 5           # Encoding dim for null tensor (A,T,G,C + gap)
    zero_pad_dim: int = 4              # Standard one-hot (A,T,G,C only)
    mismatch_rate: float = 0.05        # Probability of mismatch at each position
    positive_ratio: float = 0.3        # Fraction of positive (off-target) samples
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    output_dir: str = "./data/generated"


@dataclass
class CNNConfig:
    """1D-CNN stream hyperparameters."""
    input_channels: int = 5            # Matches null_tensor_dim
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    num_filters: int = 64              # Filters per kernel size
    dropout: float = 0.3
    use_residual: bool = True
    output_dim: int = 128              # Final CNN feature dimension


@dataclass
class TransformerConfig:
    """DNABERT-2 + LoRA parameters."""
    model_name: str = "zhihan1996/DNABERT-2-117M"
    max_length: int = 64               # Max token length
    lora_r: int = 8                    # LoRA rank
    lora_alpha: int = 16               # LoRA scaling
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["Wqkv"]  # DNABERT-2 uses fused QKV projection
    )
    output_dim: int = 128              # Projected embedding dimension
    freeze_base: bool = False          # False because we use LoRA


@dataclass
class KANConfig:
    """Kolmogorov-Arnold Network parameters."""
    input_dim: int = 256               # CNN (128) + Transformer (128)
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 1                # Binary risk score
    spline_order: int = 3              # B-spline order (cubic)
    num_knots: int = 8                 # Number of knot points per spline
    l1_penalty: float = 0.01           # L1 regularization on spline coefficients


@dataclass
class TrainingConfig:
    """Training loop parameters."""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0          # Paper says no weight decay with Adam
    focal_gamma: float = 2.0           # Focal loss focusing parameter
    lambda_focal: float = 0.7          # Weight for focal loss
    lambda_reg: float = 0.25           # Weight for spline regularization
    # Note: lambda_focal + lambda_reg < 1.0 (0.95), remaining 0.05 implicit
    scheduler: str = "cosine"          # Cosine annealing
    early_stopping_patience: int = 7
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class RAGConfig:
    """RAG + LLM module parameters."""
    vector_store: str = "chromadb"     # "chromadb" or "faiss"
    collection_name: str = "genomic_annotations"
    embedding_model: str = "all-MiniLM-L6-v2"  # For RAG retrieval
    llm_model: str = "google/flan-t5-base"      # Small LLM for Colab
    # Alternative: use Groq API or together.ai for larger models
    top_k_retrieval: int = 3
    risk_threshold: float = 0.5        # Above this → high risk → generate audit
    temperature: float = 0.3


@dataclass
class UIConfig:
    """Streamlit app parameters."""
    page_title: str = "Neuro-CRISPR-KAN | Safety Audit Dashboard"
    theme_color: str = "#1E88E5"
    show_attention_heatmap: bool = True
    show_comparison_chart: bool = True


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    kan: KANConfig = field(default_factory=KANConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# Global config instance
config = Config()


if __name__ == "__main__":
    print(f"Device: {config.device}")
    print(f"Dataset size: {config.data.num_samples}")
    print(f"KAN hidden dims: {config.kan.hidden_dims}")
    print(f"LoRA rank: {config.transformer.lora_r}")
    print(f"Focal gamma: {config.training.focal_gamma}")
