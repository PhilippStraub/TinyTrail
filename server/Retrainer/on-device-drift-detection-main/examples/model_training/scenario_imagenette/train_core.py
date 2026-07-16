#!/usr/bin/env python3
"""
Imagenette: MobileNetV2 backbone training (all 10 classes).

Imagenette is a 10-class subset of ImageNet, so we use the ImageNet-pretrained
MobileNetV2 transfer-learning path (img_size=224, MobileNetV2 preprocessing).
The exported TFLite model is the INT8 feature extractor (1280-dim feature vector).

This is the shared feature extractor for all three Imagenette classifier
scenarios (KNN / KMeans / NearestCentroid). It is trained once and the artifacts
are mirrored into the three python_examples model folders by default.

Outputs (per target model folder):
- model_object.tflite  (INT8 feature extractor, 1280-dim)
- labels_object.txt
"""

import argparse
from pathlib import Path
import sys

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent  # project root
TRAIN_DIR = ROOT_DIR / "examples/dataset_imagenette/train"
SCENARIO_DIR = Path(__file__).parent

# Default target: the KNN python example. The other two classifier scenarios
# (KMeans / NearestCentroid) reuse the *same* feature extractor, so we mirror the
# artifacts into their model folders too.
PY_EXAMPLES_DIR = ROOT_DIR / "examples/python_examples"
PY_EXAMPLE_MODEL_DIR = PY_EXAMPLES_DIR / "scenario_imagenette_knn/model"
DEFAULT_MIRROR_DIRS = [
    PY_EXAMPLES_DIR / "scenario_imagenette_kmeans/model",
    PY_EXAMPLES_DIR / "scenario_imagenette_nearest_centroid/model",
]

# Make shared training utils importable
sys.path.insert(0, str(ROOT_DIR / "examples/model_training"))
from common.training import (
    build_datasets,
    train_and_export,
    DEFAULT_LR,
    DEFAULT_FINE_TUNE_LR,
    DEFAULT_IMG_SIZE,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Imagenette backbone training (pretrained MobileNetV2, all 10 classes by default)"
    )
    p.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--model-dir", type=Path, default=PY_EXAMPLE_MODEL_DIR)
    p.add_argument("--include-classes", nargs="*", default=None)
    p.add_argument("--exclude-classes", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--fine-tune-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--fine-tune-lr", type=float, default=DEFAULT_FINE_TUNE_LR)
    # Mirror the trained artifacts into the KMeans / NearestCentroid model folders
    # by default (they share this feature extractor). Pass --no-mirror to skip.
    p.add_argument("--mirror-dir", action="append", default=None,
                   help="Extra dir to mirror artifacts into (can repeat). "
                        "If omitted, the KMeans/NearestCentroid scenarios are used.")
    p.add_argument("--no-mirror", action="store_true",
                   help="Do not mirror artifacts into the other scenario folders.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_mirror:
        mirror_dirs = []
    elif args.mirror_dir is not None:
        mirror_dirs = args.mirror_dir
    else:
        mirror_dirs = DEFAULT_MIRROR_DIRS

    print("=" * 50)
    print("Imagenette: Backbone training (pretrained MobileNetV2 feature extractor)")
    print("=" * 50)
    print(f"Data dir:   {args.data_dir}")
    print(f"Model dir:  {args.model_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Mirror to:  {[str(m) for m in mirror_dirs] or 'none'}")
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
        mirror_dirs=mirror_dirs,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        img_size=args.img_size,
    )

    print(f"\n✓ Training complete. Model: {model_path}")
    print(f"✓ Classes: {labels}")
