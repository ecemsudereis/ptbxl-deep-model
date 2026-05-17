"""Download the PTB-XL dataset (Person A).

PTB-XL is openly available via PhysioNet:
    https://physionet.org/content/ptb-xl/

Size note:
    - 100 Hz version (~1.7 GB)  ← USE THIS
    - 500 Hz version (~3 GB)    ← not needed

Citation (for the paper):
    Wagner, P., Strodthoff, N., Bousseljot, RD. et al.
    "PTB-XL, a large publicly available electrocardiography dataset."
    Scientific Data 7, 154 (2020).

This file is only a SKELETON. The code is filled in later.
Raw data lands under data/raw/ and is in .gitignore (NOT committed).
"""

from pathlib import Path

# PhysioNet PTB-XL root URL (100 Hz). Person A verifies.
PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/"
RAW_DIR = Path(__file__).parent / "raw"


def download_ptbxl(target_dir: Path = RAW_DIR, sampling_rate: int = 100) -> Path:
    """Download PTB-XL and extract it under target_dir.

    Args:
        target_dir: Folder for the raw data (default data/raw/).
        sampling_rate: 100 (default) or 500.

    Returns:
        Path to the downloaded data root.

    Expected structure (after download):
        data/raw/
        ├── ptbxl_database.csv      # meta + strat_fold + scp_codes
        ├── scp_statements.csv      # SCP code → diagnostic superclass map
        └── records100/             # WFDB .dat/.hea files (100 Hz)
    """
    # TODO(A): download via wget/requests + unzip.
    #          Prefer `wfdb.io.dl_database` or the direct PhysioNet link.
    raise NotImplementedError("Person A: download logic goes here.")


if __name__ == "__main__":
    print("download.py — skeleton. Person A fills this in.")
    print(f"Target dir: {RAW_DIR}")
