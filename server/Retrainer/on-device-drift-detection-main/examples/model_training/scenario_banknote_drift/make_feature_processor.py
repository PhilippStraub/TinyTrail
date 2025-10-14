#!/usr/bin/env python3
"""
Scenario 3: Feature processor training (includes all classes by default).
Thin wrapper around common.feature_processor with CLI parameters.
"""

import argparse
import sys
from pathlib import Path

# Paths - adjusted
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SCENARIO_DIR = Path(__file__).parent
PY_EXAMPLE_MODEL_DIR = ROOT_DIR / "examples/python_examples/scenario_banknote_drift/model"
DEFAULT_MODEL_PATH = PY_EXAMPLE_MODEL_DIR / "model_object.tflite"
DEFAULT_DATA_DIR = ROOT_DIR / "examples/data2"
DEFAULT_OUTPUT_DIR = PY_EXAMPLE_MODEL_DIR

# Import common module
sys.path.insert(0, str(ROOT_DIR / "examples/model_training"))
# Also ensure repo root for local package imports in common module
sys.path.insert(0, str(ROOT_DIR))
from common.feature_processor import (
    build_and_save_processor,
    DEFAULT_IMG_SIZE,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_MAX_SAMPLES,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train feature processor (scaler+PCA)")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--exclude-classes", nargs="*", default=None, help="Classes to exclude")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--mirror-dir", action="append", default=[], help="Mirror output here (can repeat)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Hint: run train_core.py (which writes to the Python example folder) or pass --model-path explicitly.")
        raise SystemExit(1)

    processor_path = build_and_save_processor(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        exclude_classes=args.exclude_classes,
        img_size=args.img_size,
        pca_components=args.pca_components,
        max_samples=args.max_samples,
        mirror_dirs=args.mirror_dir,
    )

    print(f"\n✓ Feature processor ready: {processor_path}")
