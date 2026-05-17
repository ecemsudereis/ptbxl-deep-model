"""Download / locate the PTB-XL dataset.

PTB-XL is openly available via PhysioNet:
    https://physionet.org/content/ptb-xl/

PhysioNet's own server was extremely slow, so we use the same dataset
mirrored on Kaggle (https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset),
downloaded manually as 'archive.zip' (~1.7 GB). The Kaggle mirror is
PTB-XL v1.0.1 -- only a few duplicate records differ from v1.0.3; the
folds and benchmark structure are identical, so it is fine for us.

Workflow:
    1. Download archive.zip from Kaggle (done manually in the browser).
    2. Put archive.zip into data/raw/.
    3. Run: python -m data.download   -> it extracts + verifies.

Raw data lands under data/raw/ and is in .gitignore (NOT committed).

Citation (for the paper):
    Wagner, P., Strodthoff, N., Bousseljot, RD. et al.
    "PTB-XL, a large publicly available electrocardiography dataset."
    Scientific Data 7, 154 (2020).
"""

from pathlib import Path
import sys
import zipfile

RAW_DIR = Path(__file__).parent / "raw"
ZIP_NAME = "archive.zip"   # the file Kaggle gives you


def find_data_root(raw_dir: Path = RAW_DIR):
    """Return the folder that directly contains ptbxl_database.csv.

    The Kaggle zip extracts into a versioned subfolder
    (e.g. ptb-xl-a-large-...-1.0.1/), so the CSVs may live either at
    data/raw/ or in that nested folder. preprocess.py reuses this
    helper so it never has to care which.
    """
    if (raw_dir / "ptbxl_database.csv").exists():
        return raw_dir
    # Search one level down for any folder that holds the CSV.
    if raw_dir.exists():
        for child in raw_dir.iterdir():
            if child.is_dir() and (child / "ptbxl_database.csv").exists():
                return child
    return None


def download_ptbxl(target_dir: Path = RAW_DIR) -> Path:
    """Extract PTB-XL from data/raw/archive.zip and verify it.

    Returns:
        Path to the folder that directly contains ptbxl_database.csv.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent: if already extracted, do nothing.
    root = find_data_root(target_dir)
    if root is not None:
        print(f"PTB-XL already extracted at: {root} -- skipping.")
        return root

    zip_path = target_dir / ZIP_NAME
    if not zip_path.exists():
        print("archive.zip not found.")
        print("STEPS:")
        print("  1. Download it from Kaggle (you already did this):")
        print("     https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset")
        print(f"  2. Move 'archive.zip' into: {target_dir.resolve()}")
        print("  3. Re-run: python -m data.download")
        sys.exit(1)

    print("Extracting archive.zip (~22k files, takes a few minutes)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    root = find_data_root(target_dir)
    if root is None:
        print("Extracted, but ptbxl_database.csv not found -- bad zip?")
        sys.exit(1)

    # Free ~1.7 GB: the extracted data is all we need from here on.
    zip_path.unlink(missing_ok=True)
    print(f"Done. Data root: {root}")
    return root


if __name__ == "__main__":
    download_ptbxl()