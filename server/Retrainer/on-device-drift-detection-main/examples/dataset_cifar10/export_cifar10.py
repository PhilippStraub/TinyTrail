"""
Export the official CIFAR-10 train/test split to per-class PNG folders.

Produces (under the scenario dataset dir, where the benchmark expects it):
    dataset_cifar10/train/<class>/train_NNNNN.png   (50,000 images, the 5 data batches)
    dataset_cifar10/test/<class>/test_NNNNN.png      (10,000 images, the test batch)

The split is the canonical CIFAR-10 split: reference samples for the KNN are drawn
from train/, and the benchmark evaluates on the held-out test/ (novelty) — never the
same images. The train_/test_ filename prefixes keep the two impossible to confuse.

WHEN to run:
    Run this ONCE before the CIFAR-10 benchmark — it downloads the archive from
    cs.toronto.edu (cached under downloads/) and unpacks it into the per-class
    PNG folders the benchmark reads from. Re-runs are cheap: a split that is
    already complete is skipped unless you pass --force.

Usage:
    python export_cifar10.py                 # export to the scenario dataset dir, skip splits already present
    python export_cifar10.py --force         # re-export even if folders exist
    python export_cifar10.py --output-dir D  # export somewhere else

See also:
    ../python_examples/scenario_cifar10_knn_benchmark/reduce_test_data.py
        Optional follow-up: thins an exported per-class folder down to a small
        random subset (e.g. 5 %) for quick smoke-test runs of the pipeline.
"""

import argparse
import pickle
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXTRACTED_DIR = "cifar-10-batches-py"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Default output = the scenario dataset dir the benchmark reads from, so both
# create_initial_state.py (train/) and the pipeline's test sampling (test/) find it.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR  # creates train/ and test/ next to this script
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"

EXPECTED_PER_CLASS = {"train": 5000, "test": 1000}


def unpickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _split_is_complete(output_dir: Path, split_name: str) -> bool:
    """True if every class folder already holds the expected image count."""
    split_dir = output_dir / split_name
    if not split_dir.is_dir():
        return False
    expected = EXPECTED_PER_CLASS[split_name]
    for cname in CLASS_NAMES:
        cdir = split_dir / cname
        if not cdir.is_dir() or sum(1 for _ in cdir.glob("*.png")) < expected:
            return False
    return True


def save_images(data_dict, output_dir: Path, split_name: str, start_idx: int = 0):
    data = data_dict[b"data"]
    labels = data_dict[b"labels"]

    # CIFAR-10: N x 3072 -> N x 3 x 32 x 32 -> N x 32 x 32 x 3 (RGB)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    for i, (img_array, label) in enumerate(zip(data, labels), start=start_idx):
        class_name = CLASS_NAMES[label]
        class_dir = output_dir / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        out_file = class_dir / f"{split_name}_{i:05d}.png"
        Image.fromarray(img_array.astype(np.uint8)).save(out_file)


def _safe_extractall(tar: tarfile.TarFile, path: Path):
    """Extract with the 'data' filter where available (py>=3.11.4) to block
    path-traversal entries; fall back to a plain extract on older runtimes."""
    try:
        tar.extractall(path=path, filter="data")
    except TypeError:
        tar.extractall(path=path)


def _count_summary(output_dir: Path, split_name: str):
    split_dir = output_dir / split_name
    print(f"  {split_name}:")
    total = 0
    for cname in CLASS_NAMES:
        n = sum(1 for _ in (split_dir / cname).glob("*.png"))
        total += n
        print(f"    {cname:<11} {n}")
    print(f"    {'TOTAL':<11} {total}")


def main():
    ap = argparse.ArgumentParser(description="Export CIFAR-10 to per-class PNG folders")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="Re-export splits even if already present")
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    archive_path = DOWNLOAD_DIR / ARCHIVE_NAME

    if not archive_path.exists():
        print("Downloading CIFAR-10 ...")
        urlretrieve(URL, archive_path)

    extracted_path = DOWNLOAD_DIR / EXTRACTED_DIR
    if not extracted_path.exists():
        print("Extracting archive ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extractall(tar, DOWNLOAD_DIR)

    print(f"Output dir: {output_dir}")

    # Train split (5 batches -> 50,000 images)
    if args.force or not _split_is_complete(output_dir, "train"):
        print("Exporting training data ...")
        train_idx = 0
        for batch_no in range(1, 6):
            batch = unpickle(extracted_path / f"data_batch_{batch_no}")
            save_images(batch, output_dir, "train", start_idx=train_idx)
            train_idx += len(batch[b"labels"])
    else:
        print("Train split already complete - skipping (use --force to redo).")

    # Test split (1 batch -> 10,000 images)
    if args.force or not _split_is_complete(output_dir, "test"):
        print("Exporting test data ...")
        test_batch = unpickle(extracted_path / "test_batch")
        save_images(test_batch, output_dir, "test", start_idx=0)
    else:
        print("Test split already complete - skipping (use --force to redo).")

    print("\nDone. Counts:")
    _count_summary(output_dir, "train")
    _count_summary(output_dir, "test")


if __name__ == "__main__":
    main()
