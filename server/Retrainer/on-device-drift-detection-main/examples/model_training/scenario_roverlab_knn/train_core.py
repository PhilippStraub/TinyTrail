#!/usr/bin/env python3
"""
Scenario autodrive3: MobileNetV2 training for autodrive3.
Thin wrapper around shared training utilities with CLI parameters.
"""

import argparse
from pathlib import Path
import sys

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent  # project root
TRAIN_DIR = ROOT_DIR / "examples/dataset_rover-lab/train"
SCENARIO_DIR = Path(__file__).parent

# Default target: the KNN python example. The other two classifier scenarios
# (KMeans / NearestCentroid) reuse the *same* feature extractor, so we mirror the
# artifacts into their model folders too.
PY_EXAMPLES_DIR = ROOT_DIR / "examples/python_examples"
PY_EXAMPLE_MODEL_DIR = PY_EXAMPLES_DIR / "scenario_rover-lab_knn/model"
DEFAULT_MIRROR_DIRS = [
    PY_EXAMPLES_DIR / "scenario_rover-lab_kmeans/model",
    PY_EXAMPLES_DIR / "scenario_rover-lab_nearest_centroid/model",
]

# Make shared training utils importable
sys.path.insert(0, str(ROOT_DIR / "examples/model_training"))
from common.training import (
    build_datasets,
    train_and_export,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FINE_TUNE_EPOCHS,
    DEFAULT_LR,
    DEFAULT_FINE_TUNE_LR,
    DEFAULT_IMG_SIZE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Scenario autodrive3 training (include all classes by default)")
    p.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    # Default: write directly into the Python example model folder
    p.add_argument("--model-dir", type=Path, default=PY_EXAMPLE_MODEL_DIR)
    p.add_argument("--include-classes", nargs="*", default=None)
    p.add_argument("--exclude-classes", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--fine-tune-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--fine-tune-lr", type=float, default=DEFAULT_FINE_TUNE_LR)
    # No mirroring by default (we already write to the Python example directory)
    p.add_argument("--mirror-dir", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("Rover-lab: Backbone training (pretrained MobileNetV2 feature extractor)")
    print("=" * 50)

    train_ds, val_ds, class_names = build_datasets(
        args.data_dir,
        include_classes=args.include_classes,
        exclude_classes=args.exclude_classes,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
        img_size=args.img_size,
    )

    model_path, labels = train_and_export(
        train_ds,
        val_ds,
        class_names,
        model_dir=args.model_dir,
        mirror_dirs=args.mirror_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        img_size=args.img_size,
    )

    print(f"\n✓ Training complete. Model: {model_path}")
    print(f"✓ Classes: {labels}")
