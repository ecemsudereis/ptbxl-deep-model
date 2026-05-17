"""PyTorch Dataset + DataLoader (Person A).

Loads the cached arrays from data/processed/ (built by preprocess.py),
splits by the official PTB-XL strat_fold (CONTRACT §3), and serves
tensors that satisfy the shape contract.

dummy=True keeps the random-tensor mode so B and C can develop in
parallel without the real data.

Contract (CONTRACT.md):
    __getitem__ ->  (signal: FloatTensor[12,1000], label: FloatTensor[5])
    get_loaders -> train/val/test DataLoader  (fold 1-8 / 9 / 10)
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocess import build_dataset

N_LEADS, SIG_LEN, N_CLASSES = 12, 1000, 5

# Official PTB-XL benchmark split (CONTRACT §3).
TRAIN_FOLDS = list(range(1, 9))   # 1..8
VAL_FOLDS = [9]
TEST_FOLDS = [10]
_SPLIT_FOLDS = {"train": TRAIN_FOLDS, "val": VAL_FOLDS, "test": TEST_FOLDS}


class PTBXLDataset(Dataset):
    """PTB-XL records for one split.

    Args:
        split: "train" | "val" | "test".
        dummy: if True, random tensors instead of real data (parallel dev).
        n_dummy: number of samples in dummy mode.
        drop_unlabeled: if True, exclude records with no superclass
            (all-zero label) -- this follows the PTB-XL benchmark, which
            evaluates only on records that have at least one diagnostic
            label.
    """

    def __init__(self, split: str = "train", dummy: bool = False,
                 n_dummy: int = 256, drop_unlabeled: bool = True):
        assert split in {"train", "val", "test"}
        self.split = split
        self.dummy = dummy
        self.n_dummy = n_dummy

        if dummy:
            return

        signals, labels, folds = build_dataset()  # cached after 1st run
        mask = np.isin(folds, _SPLIT_FOLDS[split])
        if drop_unlabeled:
            mask &= labels.sum(axis=1) > 0
        self.signals = signals[mask]
        self.labels = labels[mask]

    def __len__(self) -> int:
        return self.n_dummy if self.dummy else len(self.signals)

    def __getitem__(self, idx: int):
        if self.dummy:
            sig = torch.randn(N_LEADS, SIG_LEN, dtype=torch.float32)
            lbl = (torch.rand(N_CLASSES) > 0.7).float()
            return sig, lbl
        sig = torch.from_numpy(self.signals[idx])      # (12,1000) float32
        lbl = torch.from_numpy(self.labels[idx])       # (5,)     float32
        return sig, lbl


def get_loaders(batch_size: int = 64, dummy: bool = False,
                num_workers: int = 0, drop_unlabeled: bool = True):
    """Return a (train, val, test) DataLoader triple.

    Note: num_workers defaults to 0 on Windows (multiprocessing data
    loading is fragile there). Bump it up on Linux if you want.
    """
    def mk(split, shuffle):
        ds = PTBXLDataset(split, dummy=dummy, drop_unlabeled=drop_unlabeled)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)
    return mk("train", True), mk("val", False), mk("test", False)


if __name__ == "__main__":
    # Real data: verify shapes, split sizes, and label sanity.
    tr, va, te = get_loaders(batch_size=32, dummy=False)
    print("Split sizes  -> train:", len(tr.dataset),
          "| val:", len(va.dataset), "| test:", len(te.dataset))
    x, y = next(iter(tr))
    print("Batch shapes -> signal:", tuple(x.shape),
          "| label:", tuple(y.shape))
    assert x.shape == (32, 12, 1000) and y.shape == (32, 5)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    print("Label sums per class (this batch):",
          y.sum(0).int().tolist())
    print("OK — real PTB-XL data satisfies the CONTRACT shapes.")