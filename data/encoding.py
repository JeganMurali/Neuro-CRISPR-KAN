"""
Encoding Module
===============
Implements two encoding strategies for sgRNA-DNA pairs:

1. **Null Tensor Encoding** (Proposed):
   - 5-channel encoding: [A, T, G, C, GAP]
   - Deletions get a dedicated gap indicator [0, 0, 0, 0, 1]
   - Preserves positional indices of flanking nucleotides
   - The GAP channel is an explicit structural signal

2. **Zero-Padding Encoding** (Baseline):
   - 4-channel encoding: [A, T, G, C]
   - Deletions filled with [0, 0, 0, 0]
   - Causes positional misalignment
   - Used for ablation comparison

Both produce fixed-length tensors of shape (sequence_length, channels).
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config
from utils.helpers import NUCLEOTIDE_TO_IDX


class NullTensorEncoder:
    """
    Null Tensor Encoding (Proposed Method).

    Encodes each nucleotide as a 5D vector:
        A → [1, 0, 0, 0, 0]
        T → [0, 1, 0, 0, 0]
        G → [0, 0, 1, 0, 0]
        C → [0, 0, 0, 1, 0]
        GAP (deletion) → [0, 0, 0, 0, 1]  ← The "Null Tensor"

    For sequences shorter than target length (due to deletion),
    the gap positions are filled with the null tensor at the
    ORIGINAL deletion position, preserving flanking nucleotide positions.
    """

    def __init__(self, target_length: int = 23):
        self.target_length = target_length
        self.channels = 5  # A, T, G, C, GAP
        self.null_vector = np.array([0, 0, 0, 0, 1], dtype=np.float32)

    def encode_nucleotide(self, nuc: str) -> np.ndarray:
        """Encode a single nucleotide."""
        vec = np.zeros(self.channels, dtype=np.float32)
        if nuc in NUCLEOTIDE_TO_IDX:
            vec[NUCLEOTIDE_TO_IDX[nuc]] = 1.0
        else:
            # Unknown nucleotide → treat as gap
            vec[4] = 1.0
        return vec

    def encode_sequence(
        self,
        sequence: str,
        has_deletion: bool = False,
        deletion_pos: int = 12,
        deletion_len: int = 3
    ) -> np.ndarray:
        """
        Encode a DNA/sgRNA sequence with null tensor for deletions.

        Args:
            sequence: DNA sequence string
            has_deletion: Whether ΔF508 deletion is present
            deletion_pos: Start position of deletion
            deletion_len: Length of deletion

        Returns:
            np.ndarray of shape (target_length, 5)
        """
        encoded = np.zeros((self.target_length, self.channels), dtype=np.float32)

        if has_deletion and len(sequence) < self.target_length:
            # Insert null tensors at deletion positions
            seq_idx = 0
            for pos in range(self.target_length):
                if deletion_pos <= pos < deletion_pos + deletion_len:
                    # This is a deletion position → null tensor
                    encoded[pos] = self.null_vector
                else:
                    if seq_idx < len(sequence):
                        encoded[pos] = self.encode_nucleotide(sequence[seq_idx])
                        seq_idx += 1
        else:
            # No deletion — standard encoding
            for i, nuc in enumerate(sequence[:self.target_length]):
                encoded[i] = self.encode_nucleotide(nuc)
            # If sequence is shorter, remaining positions stay zero
            # (shouldn't happen without deletion, but safety check)

        return encoded

    def encode_pair(
        self,
        sgrna: str,
        dna: str,
        has_deletion: bool = False
    ) -> np.ndarray:
        """
        Encode an sgRNA-DNA pair.
        Stacks both encodings → shape (2, target_length, 5).
        """
        sgrna_enc = self.encode_sequence(sgrna, has_deletion=False)
        dna_enc = self.encode_sequence(
            dna, has_deletion=has_deletion,
            deletion_pos=config.data.deletion_position,
            deletion_len=config.data.deletion_length
        )
        return np.stack([sgrna_enc, dna_enc], axis=0)


class ZeroPadEncoder:
    """
    Zero-Padding Encoding (Baseline Method).

    Standard one-hot with 4 channels. Deletions are zero-padded
    at the END of the sequence, causing positional misalignment.
    Used for ablation comparison.
    """

    def __init__(self, target_length: int = 23):
        self.target_length = target_length
        self.channels = 4  # A, T, G, C only

    def encode_nucleotide(self, nuc: str) -> np.ndarray:
        vec = np.zeros(self.channels, dtype=np.float32)
        if nuc in NUCLEOTIDE_TO_IDX:
            vec[NUCLEOTIDE_TO_IDX[nuc]] = 1.0
        return vec

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode with zero-padding at end (no position preservation).
        Shape: (target_length, 4)
        """
        encoded = np.zeros((self.target_length, self.channels), dtype=np.float32)
        for i, nuc in enumerate(sequence[:self.target_length]):
            encoded[i] = self.encode_nucleotide(nuc)
        # Remaining positions (from deletion) are all zeros → noise!
        return encoded

    def encode_pair(self, sgrna: str, dna: str, **kwargs) -> np.ndarray:
        sgrna_enc = self.encode_sequence(sgrna)
        dna_enc = self.encode_sequence(dna)
        return np.stack([sgrna_enc, dna_enc], axis=0)


class CRISPRDataset(Dataset):
    """
    PyTorch Dataset for CRISPR off-target prediction.

    Returns:
        encoded_pair: Tensor of shape (2, seq_len, channels)
        off_target_label: Binary label (0 or 1)
        efficiency_score: Regression target [0, 1]
        metadata: Dict with chromatin_score, has_deletion, etc.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        encoder: str = "null_tensor",
        target_length: int = 23
    ):
        self.df = dataframe.reset_index(drop=True)

        if encoder == "null_tensor":
            self.encoder = NullTensorEncoder(target_length)
        elif encoder == "zero_pad":
            self.encoder = ZeroPadEncoder(target_length)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.encoder_name = encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Encode the sgRNA-DNA pair
        encoded = self.encoder.encode_pair(
            sgrna=row["sgrna_seq"],
            dna=row["dna_seq"],
            has_deletion=bool(row["has_deletion"])
        )

        # Convert to tensors
        encoded_tensor = torch.FloatTensor(encoded)   # (2, seq_len, channels)
        label = torch.FloatTensor([row["off_target_label"]])
        efficiency = torch.FloatTensor([row["efficiency_score"]])
        chromatin = torch.FloatTensor([row["chromatin_score"]])

        return {
            "encoded": encoded_tensor,
            "label": label,
            "efficiency": efficiency,
            "chromatin": chromatin,
            "has_deletion": torch.FloatTensor([row["has_deletion"]]),
            "sgrna_seq": row["sgrna_seq"],
            "dna_seq": row["dna_seq"],
        }


def create_dataloaders(
    df: pd.DataFrame,
    encoder: str = "null_tensor",
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> dict:
    """
    Split dataset and create DataLoaders.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    # Shuffle
    df = df.sample(frac=1, random_state=config.data.seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Encoder: {encoder}")

    loaders = {}
    for split_name, split_df, shuffle in [
        ("train", train_df, True),
        ("val", val_df, False),
        ("test", test_df, False),
    ]:
        dataset = CRISPRDataset(split_df, encoder=encoder)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Colab-safe
            drop_last=(split_name == "train"),
        )

    return loaders


if __name__ == "__main__":
    # Quick test
    from data.data_generation import generate_dataset

    df = generate_dataset()

    # Test Null Tensor encoding
    enc = NullTensorEncoder()
    sample = enc.encode_pair("ATCGATCGATCGATCGATCGNGG", "ATCGATCGATCGATCGATCGNGG")
    print(f"Null Tensor pair shape: {sample.shape}")  # (2, 23, 5)

    # Test with deletion
    sample_del = enc.encode_pair(
        "ATCGATCGATCGATCGATCGNGG",
        "ATCGATCGATCATCGATCGNGG",  # 3 nucleotides deleted
        has_deletion=True
    )
    print(f"Deleted pair shape: {sample_del.shape}")  # Still (2, 23, 5)

    # Test DataLoaders
    loaders = create_dataloaders(df, encoder="null_tensor", batch_size=32)
    batch = next(iter(loaders["train"]))
    print(f"\nBatch shapes:")
    print(f"  encoded: {batch['encoded'].shape}")  # (32, 2, 23, 5)
    print(f"  label: {batch['label'].shape}")       # (32, 1)
