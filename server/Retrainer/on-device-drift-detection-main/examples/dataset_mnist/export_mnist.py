"""
Export the MNIST train/test split to per-class PNG folders.

Produces (under the scenario dataset dir, where the benchmark expects it):
    dataset_mnist/train/<digit>/train_NNNNN.png   (60,000 images)
    dataset_mnist/test/<digit>/test_NNNNN.png      (10,000 images)

The split is the canonical MNIST split: reference samples for the KNN are drawn
from train/, and the benchmark evaluates on the held-out test/ (novelty) — never the
same images. The train_/test_ filename prefixes keep the two impossible to confuse.
This mirrors ../dataset_cifar10/export_cifar10.py.

Source: the original host (yann.lecun.com/exdb/mnist/) has become unreliable, so this
downloads the identical idx-ubyte.gz files from Google's public mirror instead
(the same files TensorFlow's own dataset utilities have historically used).

WHEN to run:
    Run this ONCE before the MNIST benchmark — it downloads the archive (cached under
    downloads/) and unpacks it into the per-class PNG folders the benchmark reads
    from. Re-runs are cheap: a split that is already complete is skipped unless you
    pass --force.

Usage:
    python export_mnist.py                 # export to the scenario dataset dir, skip splits already present
    python export_mnist.py --force         # re-export even if folders exist
    python export_mnist.py --output-dir D  # export somewhere else
"""

import argparse
import gzip
import struct
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

MIRROR_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [str(d) for d in range(10)]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR  # creates train/ and test/ next to this script
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"

EXPECTED_TOTAL = {"train": 60000, "test": 10000}


def _read_idx_images(gz_path: Path) -> np.ndarray:
    """Parse an idx3-ubyte.gz image file into an (N, H, W) uint8 array."""
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {gz_path} (expected 2051)")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _read_idx_labels(gz_path: Path) -> np.ndarray:
    """Parse an idx1-ubyte.gz label file into an (N,) uint8 array."""
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {gz_path} (expected 2049)")
        return np.frombuffer(f.read(n), dtype=np.uint8)


def _split_is_complete(output_dir: Path, split_name: str) -> bool:
    """True if every class folder exists and the split's total image count matches.

    Unlike CIFAR-10, MNIST's per-class counts are not perfectly balanced (e.g. digit
    '5' has fewer training samples than digit '1'), so completeness is checked against
    the split's known total rather than a fixed per-class count.
    """
    split_dir = output_dir / split_name
    if not split_dir.is_dir():
        return False
    total = 0
    for cname in CLASS_NAMES:
        cdir = split_dir / cname
        if not cdir.is_dir():
            return False
        total += sum(1 for _ in cdir.glob("*.png"))
    return total == EXPECTED_TOTAL[split_name]


def save_images(images: np.ndarray, labels: np.ndarray, output_dir: Path, split_name: str) -> None:
    for i, (img_array, label) in enumerate(zip(images, labels)):
        class_name = CLASS_NAMES[int(label)]
        class_dir = output_dir / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        out_file = class_dir / f"{split_name}_{i:05d}.png"
        # Single-channel grayscale PNG, matching the source data. Every downstream
        # consumer (tf.io.decode_image(channels=3) during training, cv2.imread during
        # create_initial_state.py / main.py at runtime) replicates grayscale into 3
        # channels on load, so no RGB conversion is needed here.
        Image.fromarray(img_array, mode="L").save(out_file)


def _count_summary(output_dir: Path, split_name: str) -> None:
    split_dir = output_dir / split_name
    print(f"  {split_name}:")
    total = 0
    for cname in CLASS_NAMES:
        n = sum(1 for _ in (split_dir / cname).glob("*.png"))
        total += n
        print(f"    {cname:<11} {n}")
    print(f"    {'TOTAL':<11} {total}")


def _download_if_missing(name: str) -> Path:
    dest = DOWNLOAD_DIR / name
    if not dest.exists():
        print(f"Downloading {name} ...")
        urlretrieve(f"{MIRROR_BASE_URL}/{name}", dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Export MNIST to per-class PNG folders")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--force", action="store_true",
                     help="Re-export splits even if already present")
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    print(f"Output dir: {output_dir}")

    if args.force or not _split_is_complete(output_dir, "train"):
        print("Exporting training data ...")
        images_path = _download_if_missing(FILES["train_images"])
        labels_path = _download_if_missing(FILES["train_labels"])
        images = _read_idx_images(images_path)
        labels = _read_idx_labels(labels_path)
        save_images(images, labels, output_dir, "train")
    else:
        print("Train split already complete - skipping (use --force to redo).")

    if args.force or not _split_is_complete(output_dir, "test"):
        print("Exporting test data ...")
        images_path = _download_if_missing(FILES["test_images"])
        labels_path = _download_if_missing(FILES["test_labels"])
        images = _read_idx_images(images_path)
        labels = _read_idx_labels(labels_path)
        save_images(images, labels, output_dir, "test")
    else:
        print("Test split already complete - skipping (use --force to redo).")

    print("\nDone. Counts:")
    _count_summary(output_dir, "train")
    _count_summary(output_dir, "test")


if __name__ == "__main__":
    main()
