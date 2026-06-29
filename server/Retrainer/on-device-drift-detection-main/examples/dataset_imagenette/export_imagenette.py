"""
Export the fastai Imagenette dataset to per-class image folders.

Produces (under the scenario dataset dir, where the benchmark expects it):
    dataset_imagenette/train/<class>/train_NNNNN.JPEG   (~9,469 images)
    dataset_imagenette/test/<class>/test_NNNNN.JPEG      (~3,925 images, Imagenette's "val")

Imagenette is a 10-class subset of ImageNet (https://github.com/fastai/imagenette).
The split is the canonical fastai split: reference samples for the KNN are drawn
from train/, and the benchmark evaluates on the held-out test/ (novelty) — never the
same images. The train_/test_ filename prefixes keep the two impossible to confuse.

WHEN to run:
    Run this ONCE before the Imagenette benchmark — it s the archive from
    the fastai S3 mirror (cached under downloads/) and unpacks it into the per-class
    folders the benchmark reads from. Re-runs are cheap: a split that is already
    complete is skipped unless you pass --force.

Usage:
    python export_imagenette.py                  # 320px variant, skip splits already present
    python export_imagenette.py --size 160       # smaller download (~98 MB)
    python export_imagenette.py --size full       # full-resolution (~1.5 GB)
    python export_imagenette.py --force           # re-export even if folders exist
    python export_imagenette.py --output-dir D    # export somewhere else

See also:
    export_cifar10.py   The CIFAR-10 equivalent this script mirrors.
"""

import argparse
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# fastai S3 mirror. The "size" picks the shortest-side-resized variant:
#   full -> original resolution (~1.5 GB), 320 -> shortest side 320px (~325 MB),
#   160  -> shortest side 160px (~98 MB). All share the same train/val/<wnid> layout.
SIZE_ARCHIVES = {
    "full": "imagenette2.tgz",
    "320": "imagenette2-320.tgz",
    "160": "imagenette2-160.tgz",
}
BASE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/"

# WordNet IDs (folder names inside the archive) -> readable, shell-safe class names.
WNID_TO_CLASS = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}
CLASS_NAMES = sorted(WNID_TO_CLASS.values())

# Imagenette ships a train/ and a val/ split; we expose val/ as test/ to match
# the CIFAR-10 layout (train/ for reference samples, test/ for held-out novelty).
SRC_TO_DST_SPLIT = {"train": "train", "val": "test"}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR  # creates train/ and test/ next to this script
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"


def _safe_extractall(tar: tarfile.TarFile, path: Path):
    """Extract with the 'data' filter where available (py>=3.11.4) to block
    path-traversal entries; fall back to a plain extract on older runtimes."""
    try:
        tar.extractall(path=path, filter="data")
    except TypeError:
        tar.extractall(path=path)


def _split_is_complete(output_dir: Path, dst_split: str) -> bool:
    """True if every class folder for this split already holds at least one image
    (Imagenette class counts vary, so we only require non-empty folders)."""
    split_dir = output_dir / dst_split
    if not split_dir.is_dir():
        return False
    for cname in CLASS_NAMES:
        cdir = split_dir / cname
        if not cdir.is_dir() or not any(cdir.glob("*.JPEG")):
            return False
    return True


def export_split(extracted_root: Path, output_dir: Path, src_split: str, dst_split: str):
    """Copy every <wnid>/*.JPEG from the extracted archive into output_dir/<dst_split>/<class>/."""
    for wnid, cname in WNID_TO_CLASS.items():
        src_dir = extracted_root / src_split / wnid
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Expected class dir not found in archive: {src_dir}")
        dst_dir = output_dir / dst_split / cname
        dst_dir.mkdir(parents=True, exist_ok=True)
        for i, src_file in enumerate(sorted(src_dir.glob("*.JPEG"))):
            shutil.copy2(src_file, dst_dir / f"{dst_split}_{i:05d}.JPEG")


def _count_summary(output_dir: Path, dst_split: str):
    split_dir = output_dir / dst_split
    print(f"  {dst_split}:")
    total = 0
    for cname in CLASS_NAMES:
        n = sum(1 for _ in (split_dir / cname).glob("*.JPEG"))
        total += n
        print(f"    {cname:<17} {n}")
    print(f"    {'TOTAL':<17} {total}")


def main():
    ap = argparse.ArgumentParser(description="Export Imagenette to per-class image folders")
    ap.add_argument("--size", choices=list(SIZE_ARCHIVES), default="320",
                    help="Image variant to download (default: 320)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="Re-export splits even if already present")
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    archive_name = SIZE_ARCHIVES[args.size]
    extracted_dir = archive_name[:-4]  # strip ".tgz" -> e.g. "imagenette2-320"

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    archive_path = DOWNLOAD_DIR / archive_name

    if not archive_path.exists():
        print(f"Downloading Imagenette ({args.size}) ...")
        urlretrieve(BASE_URL + archive_name, archive_path)

    extracted_path = DOWNLOAD_DIR / extracted_dir
    if not extracted_path.exists():
        print("Extracting archive ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extractall(tar, DOWNLOAD_DIR)

    print(f"Output dir: {output_dir}")

    for src_split, dst_split in SRC_TO_DST_SPLIT.items():
        if args.force or not _split_is_complete(output_dir, dst_split):
            print(f"Exporting {dst_split} data (from Imagenette '{src_split}') ...")
            export_split(extracted_path, output_dir, src_split, dst_split)
        else:
            print(f"{dst_split.capitalize()} split already complete - skipping (use --force to redo).")

    print("\nDone. Counts:")
    for dst_split in SRC_TO_DST_SPLIT.values():
        _count_summary(output_dir, dst_split)


if __name__ == "__main__":
    main()
