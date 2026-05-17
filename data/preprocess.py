"""PTB-XL preprocessing (Person A).

Pipeline (follows the official PTB-XL benchmark protocol,
Strodthoff et al. 2021):
    1. Read ptbxl_database.csv (meta + scp_codes + strat_fold).
    2. Parse scp_codes; map each record to the 5 diagnostic
       superclasses via scp_statements.csv (NORM, MI, STTC, CD, HYP).
       Multi-label: a record can have several superclasses.
    3. Read the 100 Hz waveform of each record via wfdb (filename_lr).
    4. Filter: baseline-wander removal + 50 Hz notch.
    5. Per-lead z-score normalization.
    6. Cache everything to data/processed/*.npy so training/ablation
       never re-reads 21k files.

Output contract (CONTRACT.md):
    signals (N, 12, 1000) float32
    labels  (N, 5)        float32  multi-label 0/1
    folds   (N,)          int      1..10
"""

from __future__ import annotations
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from tqdm import tqdm

from data.download import find_data_root

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]  # fixed order — CONTRACT §2
SUPER_IDX = {c: i for i, c in enumerate(SUPERCLASSES)}
TARGET_LEN = 1000   # 100 Hz × 10 s
N_LEADS = 12
FS = 100            # sampling rate of the 100 Hz records

PROCESSED_DIR = Path(__file__).parent / "processed"


# ----------------------------------------------------------------------
# 1-2. Labels: SCP codes -> 5 superclass multi-label vector
# ----------------------------------------------------------------------
def _build_scp_to_super(scp_statements: pd.DataFrame) -> dict:
    """SCP code -> superclass string, only for diagnostic statements."""
    diag = scp_statements[scp_statements["diagnostic"] == 1]
    return diag["diagnostic_class"].dropna().to_dict()


def scp_to_superclass(scp_codes: dict, scp_to_super: dict) -> np.ndarray:
    """Convert one record's scp_codes dict into a (5,) 0/1 vector.

    Args:
        scp_codes: e.g. {'NORM': 100.0, 'LVOLT': 0.0}.
        scp_to_super: SCP code -> superclass string (from helper above).

    Returns:
        (5,) float32, multi-label over NORM, MI, STTC, CD, HYP.
    """
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for code in scp_codes:                       # keys are the SCP codes
        sup = scp_to_super.get(code)
        if sup in SUPER_IDX:
            y[SUPER_IDX[sup]] = 1.0
    return y


# ----------------------------------------------------------------------
# 4. Signal filtering
# ----------------------------------------------------------------------
def filter_signal(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """Baseline-wander highpass (0.5 Hz) + 50 Hz powerline notch.

    Args:
        sig: (12, 1000) raw signal.
        fs: sampling rate.

    Returns:
        (12, 1000) filtered signal (same shape).

    Note: this is only a coarse pre-clean. The DAE block learns the
    deeper signal structure; we deliberately do NOT over-filter.
    """
    # High-pass 0.5 Hz (remove slow baseline wander), zero-phase.
    sos = butter(2, 0.5, btype="highpass", fs=fs, output="sos")
    sig = sosfiltfilt(sos, sig, axis=-1)
    # 50 Hz notch (mains interference in Europe).
    b, a = iirnotch(w0=50.0, Q=30.0, fs=fs)
    # iirnotch is IIR (b,a) -> use filtfilt for zero phase.
    from scipy.signal import filtfilt
    sig = filtfilt(b, a, sig, axis=-1)
    return sig.astype(np.float32)


# ----------------------------------------------------------------------
# 5. Normalization
# ----------------------------------------------------------------------
def normalize(sig: np.ndarray) -> np.ndarray:
    """Per-lead z-score. (12,1000) -> (12,1000). Guards std=0."""
    mean = sig.mean(axis=-1, keepdims=True)
    std = sig.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)         # avoid divide-by-zero
    return ((sig - mean) / std).astype(np.float32)


# ----------------------------------------------------------------------
# 3-6. Full pipeline + cache
# ----------------------------------------------------------------------
def build_dataset(force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the whole pipeline, cache to data/processed/, return arrays.

    Args:
        force: if True, rebuild even if a cache exists.

    Returns:
        signals (N,12,1000) float32, labels (N,5) float32, folds (N,) int
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sig_p = PROCESSED_DIR / "signals.npy"
    lbl_p = PROCESSED_DIR / "labels.npy"
    fold_p = PROCESSED_DIR / "folds.npy"

    if not force and sig_p.exists() and lbl_p.exists() and fold_p.exists():
        print("Loading cached processed data from data/processed/ ...")
        return (np.load(sig_p), np.load(lbl_p), np.load(fold_p))

    root = find_data_root()
    if root is None:
        raise FileNotFoundError(
            "PTB-XL not found. Run `python -m data.download` first."
        )

    print("Reading metadata...")
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_stmt = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp_to_super = _build_scp_to_super(scp_stmt)

    print(f"Reading + processing {len(df)} ECG records (100 Hz)...")
    signals, labels, folds = [], [], []
    for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
        # filename_lr is like 'records100/00000/00001_lr' (no extension)
        rec_path = root / row["filename_lr"]
        sig, _ = wfdb.rdsamp(str(rec_path))      # (1000, 12)
        sig = sig.T.astype(np.float32)           # -> (12, 1000)

        sig = filter_signal(sig)
        sig = normalize(sig)

        signals.append(sig)
        labels.append(scp_to_superclass(row["scp_codes"], scp_to_super))
        folds.append(int(row["strat_fold"]))

    signals = np.stack(signals).astype(np.float32)   # (N,12,1000)
    labels = np.stack(labels).astype(np.float32)     # (N,5)
    folds = np.array(folds, dtype=np.int64)          # (N,)

    np.save(sig_p, signals)
    np.save(lbl_p, labels)
    np.save(fold_p, folds)
    print(f"Cached to {PROCESSED_DIR}")
    print(f"signals {signals.shape} | labels {labels.shape} "
          f"| folds {folds.shape}")
    return signals, labels, folds


if __name__ == "__main__":
    s, y, f = build_dataset()
    print("\n--- Sanity check ---")
    print("signals:", s.shape, s.dtype, "| labels:", y.shape,
          "| folds:", f.shape)
    print("Per-superclass positive counts:",
          dict(zip(SUPERCLASSES, y.sum(axis=0).astype(int))))
    print("Records with >=1 label:", int((y.sum(1) > 0).sum()), "/", len(y))