"""
Pull the rover-lab image pool from its public Hugging Face dataset repo and split it
into the train/test layout the benchmark reads from.

Produces (under the scenario dataset dir, where the benchmark expects it):
    dataset_rover-lab/train/<label>/train_NNNNN.jpg
    dataset_rover-lab/test/<label>/test_NNNNN.jpg

The uploaded archive is a flat, unsplit pool (images/rover_NNNNN.jpg + labels.csv, built
by prepare_rover_lab_archive.py) — deliberately without a baked-in train/test split. This
script performs that split itself, deterministically: per label, it sorts the pool's
filenames and shuffles them with `random.Random(seed + label_index)`, then draws
`test_fraction` of them as the held-out test set. Same seed + same pool -> same split on
every machine, so the split never needs to be re-uploaded or shipped as a second artifact.

WHEN to run:
    Run this ONCE before the rover-lab benchmark on a machine that doesn't already have
    the split — it downloads the archive from Hugging Face (cached under downloads/)
    and builds the per-class train/test folders the benchmark reads from. Re-runs are
    cheap: a split that is already complete is skipped unless you pass --force.

Usage:
    python export_rover-lab.py                          # default seed=42, test_fraction=0.2
    python export_rover-lab.py --force                   # re-export even if folders exist
    python export_rover-lab.py --seed 123 --test-fraction 0.3
    python export_rover-lab.py --output-dir D

See also:
    prepare_rover_lab_archive.py   Builds the flat pool + archive this script downloads.
"""

import argparse
import csv
import random
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# Replace with your own Hugging Face dataset repo (created via prepare_rover_lab_archive.py
# + `huggingface-cli upload <namespace>/rover-lab rover-lab.tar.gz`).
HF_REPO = "Fundulus/rover-lab"
ARCHIVE_NAME = "rover-lab.tar.gz"
BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR  # creates train/ and test/ next to this script
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"
POOL_DIR = DOWNLOAD_DIR / "pool"  # extracted images/ + labels.csv live here


def _safe_extractall(tar: tarfile.TarFile, path: Path):
    """Extract with the 'data' filter where available (py>=3.11.4) to block
    path-traversal entries; fall back to a plain extract on older runtimes."""
    try:
        tar.extractall(path=path, filter="data")
    except TypeError:
        tar.extractall(path=path)


def _load_labels(pool_dir: Path) -> dict[str, list[str]]:
    """Return {label: [file_name, ...]} sorted by file_name for reproducible ordering."""
    by_label: dict[str, list[str]] = {}
    with open(pool_dir / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_label.setdefault(row["label"], []).append(row["file_name"])
    for label in by_label:
        by_label[label].sort()
    return by_label


def _split_is_complete(output_dir: Path, labels: list[str]) -> bool:
    for split in ("train", "test"):
        split_dir = output_dir / split
        if not split_dir.is_dir():
            return False
        for label in labels:
            if not any((split_dir / label).glob("*.jpg")):
                return False
    return True


def build_split(pool_dir: Path, output_dir: Path, seed: int, test_fraction: float) -> None:
    by_label = _load_labels(pool_dir)
    images_dir = pool_dir / "images"

    for li, (label, file_names) in enumerate(sorted(by_label.items())):
        rng = random.Random(seed + li)
        shuffled = list(file_names)
        rng.shuffle(shuffled)

        n_test = max(1, round(len(shuffled) * test_fraction))
        test_files = shuffled[:n_test]
        train_files = shuffled[n_test:]

        for split_name, files in (("train", train_files), ("test", test_files)):
            out_class = output_dir / split_name / label
            out_class.mkdir(parents=True, exist_ok=True)
            for i, file_name in enumerate(files):
                dst_file = out_class / f"{split_name}_{i:05d}.jpg"
                dst_file.write_bytes((images_dir / file_name).read_bytes())

        print(f"  {label:<12} train={len(train_files):<5} test={len(test_files)}")


def _count_summary(output_dir: Path, split_name: str, labels: list[str]):
    split_dir = output_dir / split_name
    print(f"  {split_name}:")
    total = 0
    for label in labels:
        n = sum(1 for _ in (split_dir / label).glob("*.jpg"))
        total += n
        print(f"    {label:<12} {n}")
    print(f"    {'TOTAL':<12} {total}")


def main():
    ap = argparse.ArgumentParser(description="Export rover-lab to a reproducible train/test split")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--force", action="store_true",
                    help="Re-export/re-split even if already present")
    args = ap.parse_args()

    if HF_REPO.startswith("<"):
        raise SystemExit(
            "Set HF_REPO in export_rover-lab.py to your Hugging Face dataset repo "
            "(e.g. 'philipp/rover-lab') before running this script."
        )

    output_dir = args.output_dir.resolve()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DOWNLOAD_DIR / ARCHIVE_NAME

    if not archive_path.exists():
        print(f"Downloading rover-lab from {BASE_URL + ARCHIVE_NAME} ...")
        urlretrieve(BASE_URL + ARCHIVE_NAME, archive_path)

    if not (POOL_DIR / "labels.csv").exists():
        print("Extracting pool (images/ + labels.csv) ...")
        POOL_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extractall(tar, POOL_DIR)

    labels = sorted(_load_labels(POOL_DIR).keys())
    print(f"Output dir: {output_dir}")

    if args.force or not _split_is_complete(output_dir, labels):
        print(f"Building train/test split (seed={args.seed}, test_fraction={args.test_fraction}) ...")
        build_split(POOL_DIR, output_dir, args.seed, args.test_fraction)
    else:
        print("Train/test split already complete - skipping (use --force to redo).")

    print("\nDone. Counts:")
    _count_summary(output_dir, "train", labels)
    _count_summary(output_dir, "test", labels)


if __name__ == "__main__":
    main()
