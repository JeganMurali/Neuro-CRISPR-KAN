"""
Synthetic Data Generation
=========================
Generates 10,000 sgRNA-DNA pairs with biologically realistic rules:
- ΔF508 deletion patterns (CTT deletion at position 1521-1523 of CFTR)
- Mismatch injection with position-dependent probability
- PAM site (NGG) enforcement
- Chromatin accessibility scores
- Class-balanced off-target labels

Output: CSV with columns:
  sgRNA_seq, dna_seq, has_deletion, num_mismatches, mismatch_positions,
  chromatin_score, pam_intact, seed_mismatches, off_target_label, efficiency_score
"""

import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import config
from utils.helpers import set_seed, NUCLEOTIDES, NUCLEOTIDE_TO_IDX, SEED_REGION


def generate_random_sequence(length: int = 23) -> str:
    """Generate a random DNA sequence of given length."""
    return "".join(random.choices(NUCLEOTIDES, k=length))


def add_pam_site(sequence: str) -> str:
    """
    Ensure the last 3 bases form a valid NGG PAM site.
    PAM is at positions 21-23 (0-indexed: 20-22) of the 23bp target.
    """
    pam_base = random.choice(NUCLEOTIDES)  # N can be anything
    return sequence[:-3] + pam_base + "GG"


def inject_mismatches(
    sgrna: str,
    dna: str,
    mismatch_rate: float = 0.15,
    seed_weight: float = 2.5
) -> Tuple[str, List[int]]:
    """
    Inject mismatches into DNA sequence relative to sgRNA.
    Mismatches in the seed region (positions 1-12 from PAM) are
    weighted more heavily as they have greater biological impact.

    Returns:
        Modified DNA sequence, list of mismatch positions
    """
    dna_list = list(dna)
    mismatch_positions = []

    for i in range(len(sgrna) - 3):  # Exclude PAM positions
        # Seed region has higher mismatch probability
        is_seed = SEED_REGION[0] <= (len(sgrna) - 3 - i) <= SEED_REGION[1]
        effective_rate = mismatch_rate * seed_weight if is_seed else mismatch_rate

        if random.random() < effective_rate:
            # Replace with a DIFFERENT nucleotide
            original = sgrna[i]
            alternatives = [n for n in NUCLEOTIDES if n != original]
            dna_list[i] = random.choice(alternatives)
            mismatch_positions.append(i)

    return "".join(dna_list), mismatch_positions


def apply_deletion(
    sequence: str,
    position: int = 12,
    length: int = 3
) -> Tuple[str, bool]:
    """
    Apply ΔF508-style deletion (remove `length` nucleotides at `position`).
    The deletion is in the DNA target, simulating the genomic mutation.

    Returns:
        Modified sequence (shorter by `length`), deletion flag
    """
    if position + length > len(sequence):
        return sequence, False
    deleted = sequence[:position] + sequence[position + length:]
    return deleted, True


def compute_chromatin_score() -> float:
    """
    Generate synthetic chromatin accessibility score.
    Range [0, 1]: 0 = closed chromatin, 1 = fully accessible.
    Real data would come from ATAC-seq or DNase-seq.
    """
    # Beta distribution gives realistic bimodal accessibility
    return round(np.random.beta(2.0, 5.0), 4)


def compute_off_target_label(
    num_mismatches: int,
    seed_mismatches: int,
    has_deletion: bool,
    chromatin_score: float,
    pam_intact: bool
) -> int:
    """
    Rule-based off-target label assignment.
    Combines multiple biological factors:
    - More mismatches → lower off-target risk
    - Seed region mismatches → much lower risk
    - PAM disruption → very low risk
    - Open chromatin → higher risk
    - Deletion presence → complex effect on risk

    Returns: 1 (off-target active) or 0 (safe)
    """
    # Base risk starts from mismatch count
    if num_mismatches == 0:
        base_risk = 0.95  # Perfect match = very high off-target risk
    elif num_mismatches <= 2:
        base_risk = 0.6
    elif num_mismatches <= 4:
        base_risk = 0.25
    else:
        base_risk = 0.05

    # Seed mismatches strongly reduce risk
    seed_penalty = 0.15 * seed_mismatches
    base_risk = max(0, base_risk - seed_penalty)

    # PAM disruption nearly eliminates risk
    if not pam_intact:
        base_risk *= 0.1

    # Open chromatin increases accessibility → higher risk
    base_risk *= (0.5 + 0.5 * chromatin_score)

    # Deletion adds complexity — can increase or decrease
    if has_deletion:
        base_risk *= random.uniform(0.7, 1.3)

    # Clamp and binarize with noise
    base_risk = np.clip(base_risk, 0, 1)
    noise = np.random.normal(0, 0.08)
    return int((base_risk + noise) > 0.35)


def compute_efficiency_score(
    num_mismatches: int,
    chromatin_score: float,
    pam_intact: bool
) -> float:
    """
    Synthetic on-target efficiency score (regression target).
    Range [0, 1]: higher = more efficient cleavage.
    """
    base = 0.8 if pam_intact else 0.1
    mismatch_penalty = 0.08 * num_mismatches
    chromatin_boost = 0.15 * chromatin_score
    noise = np.random.normal(0, 0.03)
    return round(np.clip(base - mismatch_penalty + chromatin_boost + noise, 0, 1), 4)


def generate_dataset(cfg=None) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Returns:
        DataFrame with all features and labels
    """
    if cfg is None:
        cfg = config.data

    set_seed(cfg.seed)
    records = []

    for i in range(cfg.num_samples):
        # 1. Generate sgRNA (the guide)
        sgrna = generate_random_sequence(cfg.sequence_length)
        sgrna = add_pam_site(sgrna)

        # 2. Create DNA target (start as perfect match, then modify)
        dna = sgrna  # Start identical

        # 3. Decide if this sample has the ΔF508 deletion
        has_deletion = random.random() < 0.4  # 40% of samples have deletion
        if has_deletion:
            dna, deletion_applied = apply_deletion(
                dna, cfg.deletion_position, cfg.deletion_length
            )
            has_deletion = deletion_applied

        # 4. Inject mismatches
        dna_with_mm, mm_positions = inject_mismatches(
            sgrna, dna, cfg.mismatch_rate
        )

        # 5. Count seed region mismatches
        seed_mm = sum(
            1 for pos in mm_positions
            if SEED_REGION[0] <= (cfg.sequence_length - 3 - pos) <= SEED_REGION[1]
        )

        # 6. Check if PAM is intact
        pam_intact = dna_with_mm[-2:] == "GG" if len(dna_with_mm) >= 2 else False

        # 7. Chromatin accessibility
        chromatin = compute_chromatin_score()

        # 8. Assign off-target label
        label = compute_off_target_label(
            len(mm_positions), seed_mm, has_deletion, chromatin, pam_intact
        )

        # 9. Efficiency score
        efficiency = compute_efficiency_score(
            len(mm_positions), chromatin, pam_intact
        )

        records.append({
            "sample_id": f"SAMPLE_{i:05d}",
            "sgrna_seq": sgrna,
            "dna_seq": dna_with_mm,
            "has_deletion": int(has_deletion),
            "num_mismatches": len(mm_positions),
            "mismatch_positions": str(mm_positions),
            "seed_mismatches": seed_mm,
            "pam_intact": int(pam_intact),
            "chromatin_score": chromatin,
            "off_target_label": label,
            "efficiency_score": efficiency,
        })

    df = pd.DataFrame(records)

    # Print dataset statistics
    print(f"\n{'='*50}")
    print(f"DATASET GENERATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(df)}")
    print(f"Positive (off-target): {df['off_target_label'].sum()} "
          f"({100*df['off_target_label'].mean():.1f}%)")
    print(f"Negative (safe): {(1 - df['off_target_label']).sum():.0f} "
          f"({100*(1-df['off_target_label'].mean()):.1f}%)")
    print(f"Samples with deletion: {df['has_deletion'].sum()} "
          f"({100*df['has_deletion'].mean():.1f}%)")
    print(f"Avg mismatches: {df['num_mismatches'].mean():.2f}")
    print(f"Avg chromatin: {df['chromatin_score'].mean():.3f}")
    print(f"Avg efficiency: {df['efficiency_score'].mean():.3f}")
    print(f"{'='*50}\n")

    return df


def save_dataset(df: pd.DataFrame, output_dir: str = None):
    """Save dataset to CSV."""
    if output_dir is None:
        output_dir = config.data.output_dir
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "crispr_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    return path


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print(df.head(10))
    print(f"\nColumn dtypes:\n{df.dtypes}")
