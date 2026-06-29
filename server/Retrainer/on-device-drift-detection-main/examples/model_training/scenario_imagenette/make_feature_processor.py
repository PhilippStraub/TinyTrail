#!/usr/bin/env python3
"""
Imagenette: feature processor training (StandardScaler + PCA), all classes by default.
Thin wrapper around common.feature_processor with CLI parameters.

IMPORTANT: Imagenette uses the pretrained MobileNetV2 feature extractor, so the
preprocessing is "mobilenetv2" ((x / 127.5) - 1.0) at img_size=224 -- this MUST
match train_core.py and the runtime/client preprocessing, otherwise the PCA is
fitted on a different feature distribution than the one seen at inference time.

The processor is trained against the KNN scenario model by default and mirrored
into the KMeans / NearestCentroid model folders (they share the feature extractor).
"""

import argparse
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SCENARIO_DIR = Path(__file__).parent

PY_EXAMPLES_DIR = ROOT_DIR / "examples/python_examples"
PY_EXAMPLE_MODEL_DIR = PY_EXAMPLES_DIR / "scenario_imagenette_knn/model"
DEFAULT_MODEL_PATH = PY_EXAMPLE_MODEL_DIR / "model_object.tflite"
# Imagenette class folders live under train/ and match train_core.py defaults
DEFAULT_DATA_DIR = ROOT_DIR / "examples/dataset_imagenette/train"
DEFAULT_OUTPUT_DIR = PY_EXAMPLE_MODEL_DIR
DEFAULT_MIRROR_DIRS = [
    PY_EXAMPLES_DIR / "scenario_imagenette_kmeans/model",
    PY_EXAMPLES_DIR / "scenario_imagenette_nearest_centroid/model",
]

# Import common module
sys.path.insert(0, str(ROOT_DIR / "examples/model_training"))
# Also ensure repo root for local package imports in common module
sys.path.insert(0, str(ROOT_DIR))
from common.feature_processor import (
    build_and_save_processor,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_MAX_SAMPLES,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train feature processor (scaler+PCA) for Imagenette")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--exclude-classes", nargs="*", default=None, help="Classes to exclude")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--preprocessing", choices=["mobilenetv2", "lowres"], default="mobilenetv2",
                   help="Must match the trained model (Imagenette = mobilenetv2)")
    p.add_argument("--mirror-dir", action="append", default=None,
                   help="Mirror output here (can repeat). If omitted, the "
                        "KMeans/NearestCentroid scenarios are used.")
    p.add_argument("--no-mirror", action="store_true",
                   help="Do not mirror the processor into the other scenario folders.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_mirror:
        mirror_dirs = []
    elif args.mirror_dir is not None:
        mirror_dirs = args.mirror_dir
    else:
        mirror_dirs = DEFAULT_MIRROR_DIRS

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
        mirror_dirs=mirror_dirs,
        preprocessing=args.preprocessing,
    )

    print(f"\n✓ Feature processor ready: {processor_path}")
