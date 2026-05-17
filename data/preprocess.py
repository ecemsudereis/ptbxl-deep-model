"""PTB-XL preprocessing (Person A).

Responsibilities:
    1. Read ptbxl_database.csv → 12-lead signal + meta per record
    2. Map scp_codes → 5 diagnostic superclasses (via scp_statements.csv)
       Classes: NORM, MI, STTC, CD, HYP  (see CONTRACT.md §2)
    3. Signal filtering (baseline wander + powerline noise)
    4. Per-lead normalization (z-score)
    5. Carry the strat_fold column as-is (for fold split — CONTRACT §3)

Output contract:
    signal:  np.ndarray  (N, 12, 1000)  float32
    labels:  np.ndarray  (N, 5)         float32  (multi-label 0/1)
    folds:   np.ndarray  (N,)           int      (1..10)

This file is a SKELETON.
"""

import numpy as np

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]  # fixed order — CONTRACT §2
TARGET_LEN = 1000   # 100 Hz × 10 s
N_LEADS = 12


def scp_to_superclass(scp_codes: dict, scp_statements) -> np.ndarray:
    """Convert a record's SCP code dict into a 5-dim multi-label vector.

    Args:
        scp_codes: dict like {"IMI": 100.0, "NDT": 0.0, ...}.
        scp_statements: scp_statements.csv (contains diagnostic_class).

    Returns:
        (5,) float32 — NORM, MI, STTC, CD, HYP as 0/1 in order.
    """
    # TODO(A): match via diagnostic_class in scp_statements, multi-label it.
    raise NotImplementedError("Person A: SCP → superclass mapping.")


def filter_signal(sig: np.ndarray, fs: int = 100) -> np.ndarray:
    """Remove baseline wander + powerline noise.

    Args:
        sig: (12, 1000) raw signal.
        fs: sampling rate.

    Returns:
        (12, 1000) filtered signal.

    Note: the DAE's job is to learn signal structure; classic filtering
    is only a coarse pre-clean. Do not over-filter (leave work to the DAE).
    """
    # TODO(A): bandpass + notch (50 Hz) via scipy.signal.
    raise NotImplementedError("Person A: signal filtering.")


def normalize(sig: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalization. (12,1000) -> (12,1000)."""
    # TODO(A): per-lead (mean, std). Add a guard for std=0.
    raise NotImplementedError("Person A: normalization.")


def build_dataset(raw_dir, sampling_rate: int = 100):
    """Run the full pipeline → return (signals, labels, folds).

    Returns:
        signals (N,12,1000) float32, labels (N,5) float32, folds (N,) int
    """
    # TODO(A): read records via wfdb, apply the steps above.
    raise NotImplementedError("Person A: full pipeline.")


if __name__ == "__main__":
    print("preprocess.py — skeleton. Class order:", SUPERCLASSES)
