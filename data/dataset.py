"""PyTorch Dataset + DataLoader (Person A).

This lets teams B and C work WITHOUT waiting for the real data:
`PTBXLDataset(dummy=True)` yields random but correctly-shaped tensors.
Once the real pipeline is ready, the same interface works with
`dummy=False`.

Contract (CONTRACT.md):
    __getitem__ ->  (signal: FloatTensor[12,1000], label: FloatTensor[5])
    get_loaders -> train/val/test DataLoader  (fold 1-8 / 9 / 10)
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader

N_LEADS, SIG_LEN, N_CLASSES = 12, 1000, 5


class PTBXLDataset(Dataset):
    """PTB-XL records.

    Args:
        split: "train" | "val" | "test".
        dummy: if True, random tensors instead of real data (parallel dev).
        n_dummy: number of samples in dummy mode.
    """

    def __init__(self, split: str = "train", dummy: bool = False, n_dummy: int = 256):
        assert split in {"train", "val", "test"}
        self.split = split
        self.dummy = dummy
        self.n_dummy = n_dummy
        if not dummy:
            # TODO(A): load preprocess.build_dataset output,
            #          filter the split by folds (CONTRACT §3).
            raise NotImplementedError("Person A: real data loading.")

    def __len__(self) -> int:
        return self.n_dummy if self.dummy else self._n_real()

    def __getitem__(self, idx: int):
        if self.dummy:
            sig = torch.randn(N_LEADS, SIG_LEN, dtype=torch.float32)
            lbl = (torch.rand(N_CLASSES) > 0.7).float()
            return sig, lbl
        # TODO(A): return real (signal, label) — same shape.
        raise NotImplementedError("Person A: real __getitem__.")

    def _n_real(self) -> int:
        raise NotImplementedError("Person A.")


def get_loaders(batch_size: int = 64, dummy: bool = False, num_workers: int = 2):
    """Return a train/val/test DataLoader triple.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    mk = lambda s, sh: DataLoader(
        PTBXLDataset(s, dummy=dummy), batch_size=batch_size,
        shuffle=sh, num_workers=num_workers,
    )
    return mk("train", True), mk("val", False), mk("test", False)


if __name__ == "__main__":
    # Verify the shape contract in dummy mode — anyone can run this.
    tr, va, te = get_loaders(batch_size=8, dummy=True, num_workers=0)
    x, y = next(iter(tr))
    print("signal batch:", tuple(x.shape), "| label batch:", tuple(y.shape))
    assert x.shape == (8, 12, 1000) and y.shape == (8, 5), "SHAPE CONTRACT VIOLATION"
    print("OK — CONTRACT shapes hold.")
