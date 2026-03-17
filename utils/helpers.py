"""
Utility Helpers
===============
Common utilities: seed setting, device management, logging setup.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def setup_logging(log_dir: str = "./logs", level=logging.INFO) -> logging.Logger:
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("neuro_crispr_kan")
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


# Nucleotide constants
NUCLEOTIDES = ["A", "T", "G", "C"]
NUCLEOTIDE_TO_IDX = {"A": 0, "T": 1, "G": 2, "C": 3}
PAM_SEQUENCE = "NGG"  # Cas9 PAM
SEED_REGION = (1, 12)  # Positions 1-12 from PAM (proximal seed)
