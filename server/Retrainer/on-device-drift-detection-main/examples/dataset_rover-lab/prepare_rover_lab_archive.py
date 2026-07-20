"""
Rebuild rover-lab as a flat, relabeled image pool ready for public upload.

Produces (next to this script):
    dataset_rover-lab/images/rover_NNNNN.jpg    (every source image, renamed, flattened)
    dataset_rover-lab/labels.csv                # file_name,label
    dataset_rover-lab/rover-lab.tar.gz          # images/ + labels.csv, ready to upload

Source classes are auto-discovered as every subfolder of this script's directory that
is not itself images/ or downloads/ (currently: negative/, obstacle/). Add a new class
simply by dropping another folder of images next to those two.

No train/test split happens here — the upload is one flat, unsplit pool. The split is
instead performed reproducibly at download time by export_rover-lab.py (fixed seed,
per-label 80/20 draw), so every machine that runs the export script ends up with the
identical train/test assignment without needing a second, pre-split artifact.

Images are renamed in a fixed, deterministic order (sorted by class, then original
filename) so re-running this script reproduces byte-identical file names.

Usage:
    python prepare_rover_lab_archive.py          # build images/ + labels.csv + archive
    python prepare_rover_lab_archive.py --force  # rebuild even if present

After it finishes, upload the printed archive to a public Hugging Face dataset repo,
then update HF_REPO in export_rover-lab.py to match.
"""

import argparse
import csv
import tarfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_NAME = "rover-lab.tar.gz"
RESERVED_DIRS = {"images", "downloads"}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def _discover_classes(root: Path) -> list[str]:
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name not in RESERVED_DIRS and not p.name.startswith(".")
    )


def _source_images(root: Path, class_names: list[str]) -> list[tuple[str, Path]]:
    """Return (label, path) pairs sorted by (class, original filename) for reproducible numbering."""
    pairs = []
    for cname in class_names:
        files = sorted(
            p for p in (root / cname).iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        pairs.extend((cname, p) for p in files)
    return pairs


def _pool_is_complete(root: Path, expected_count: int) -> bool:
    images_dir = root / "images"
    labels_csv = root / "labels.csv"
    if not images_dir.is_dir() or not labels_csv.is_file():
        return False
    return sum(1 for _ in images_dir.glob("rover_*.jpg")) == expected_count


def build_pool(root: Path, class_names: list[str]) -> None:
    images_dir = root / "images"
    images_dir.mkdir(exist_ok=True)
    pairs = _source_images(root, class_names)

    with open(root / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label"])
        for idx, (label, src_file) in enumerate(pairs):
            file_name = f"rover_{idx:05d}.jpg"
            (images_dir / file_name).write_bytes(src_file.read_bytes())
            writer.writerow([file_name, label])

    counts: dict[str, int] = {}
    for label, _ in pairs:
        counts[label] = counts.get(label, 0) + 1
    for cname in class_names:
        print(f"  {cname:<12} {counts.get(cname, 0)}")
    print(f"  {'TOTAL':<12} {len(pairs)}")


def pack_archive(root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(root / "images", arcname="images")
        tar.add(root / "labels.csv", arcname="labels.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the flat rover-lab image pool + upload archive")
    ap.add_argument("--force", action="store_true", help="Rebuild pool + archive even if present")
    args = ap.parse_args()

    class_names = _discover_classes(SCRIPT_DIR)
    if not class_names:
        raise SystemExit(f"No class folders found next to {__file__}")
    print("Classes:", class_names)

    expected_count = len(_source_images(SCRIPT_DIR, class_names))

    if args.force or not _pool_is_complete(SCRIPT_DIR, expected_count):
        print("Building images/ + labels.csv ...")
        build_pool(SCRIPT_DIR, class_names)
    else:
        print("images/ + labels.csv already complete - skipping (use --force to redo).")

    archive_path = SCRIPT_DIR / ARCHIVE_NAME
    if args.force or not archive_path.exists():
        print(f"Packing {archive_path.name} ...")
        pack_archive(SCRIPT_DIR, archive_path)
    else:
        print(f"{archive_path.name} already exists - skipping (use --force to redo).")

    print(f"\nDone. Upload this file to your Hugging Face dataset repo:\n  {archive_path}")
    print("Then point export_rover-lab.py's HF_REPO at your dataset repo.")


if __name__ == "__main__":
    main()
