#!/usr/bin/env python3
"""
Scenario CIFAR-10: Low-res MobileNet training for CIFAR-10 benchmark.

This script can train either:
- lowres: new CIFAR10 Low-res MobileNet
- old: old ImageNet MobileNetV2 transfer-learning model
"""

import argparse
from pathlib import Path
import sys

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent  # project root
TRAIN_DIR = ROOT_DIR / "examples/dataset_cifar10/train"

SCENARIO_DIR = Path(__file__).parent
PY_EXAMPLE_MODEL_DIR = (
    ROOT_DIR / "examples/python_examples/scenario_cifar10_knn_benchmark/model"
)

# Make shared training utils importable
sys.path.insert(0, str(ROOT_DIR / "examples/model_training"))

from common.training import (
    # old MobileNetV2 training
    build_datasets,
    train_and_export,
    DEFAULT_LR,
    DEFAULT_FINE_TUNE_LR,

    # new CIFAR10 / Low-res MobileNet training
    build_datasets_cifar10_lowres,
    train_and_export_cifar10_lowres,
    CIFAR10_BATCH_SIZE,
    CIFAR10_EPOCHS,
    CIFAR10_LR,
    CIFAR10_IMG_SIZE,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="CIFAR-10 training: Low-res MobileNet or old MobileNetV2"
    )

    p.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--model-dir", type=Path, default=PY_EXAMPLE_MODEL_DIR)

    p.add_argument("--include-classes", nargs="*", default=None)
    p.add_argument("--exclude-classes", nargs="*", default=None)

    p.add_argument(
        "--model-type",
        choices=["lowres", "old"],
        default="lowres",
        help="lowres = new CIFAR10 Low-res MobileNet, old = old MobileNetV2",
    )

    p.add_argument("--batch-size", type=int, default=CIFAR10_BATCH_SIZE)
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=CIFAR10_IMG_SIZE)

    # For quick tests you can pass: --epochs 8
    p.add_argument("--epochs", type=int, default=CIFAR10_EPOCHS)

    # Only used for old MobileNetV2
    p.add_argument("--fine-tune-epochs", type=int, default=5)

    p.add_argument("--lr", type=float, default=CIFAR10_LR)
    p.add_argument("--fine-tune-lr", type=float, default=DEFAULT_FINE_TUNE_LR)

    # No mirroring by default
    p.add_argument("--mirror-dir", action="append", default=[])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("CIFAR-10 training")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Model dir:  {args.model_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs:     {args.epochs}")
    print("=" * 60)

    if args.model_type == "lowres":
        train_ds, val_ds, class_names = build_datasets_cifar10_lowres(
            args.data_dir,
            include_classes=args.include_classes,
            exclude_classes=args.exclude_classes,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            seed=args.seed,
            img_size=args.img_size,
        )

        model_path, labels = train_and_export_cifar10_lowres(
            train_ds=train_ds,
            val_ds=val_ds,
            class_names=class_names,
            model_dir=args.model_dir,
            mirror_dirs=args.mirror_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            img_size=args.img_size,
        )

    else:
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
            train_ds=train_ds,
            val_ds=val_ds,
            class_names=class_names,
            model_dir=args.model_dir,
            mirror_dirs=args.mirror_dir,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            learning_rate=DEFAULT_LR,
            fine_tune_lr=args.fine_tune_lr,
            img_size=args.img_size,
        )

    print(f"\n✓ Training complete. Model: {model_path}")
    print(f"✓ Classes: {labels}")