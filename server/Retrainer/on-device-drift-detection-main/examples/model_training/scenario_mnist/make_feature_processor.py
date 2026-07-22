#!/usr/bin/env python3
"""
MNIST: Feature processor training (includes all classes by default).
Thin wrapper around common.feature_processor with CLI parameters.

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
PY_EXAMPLE_MODEL_DIR = PY_EXAMPLES_DIR / "scenario_mnist_knn/model"
DEFAULT_MODEL_PATH = PY_EXAMPLE_MODEL_DIR / "model_object.tflite"
# MNIST digit folders live under train/ and match train_core.py defaults
DEFAULT_DATA_DIR = ROOT_DIR / "examples/dataset_mnist/train"
DEFAULT_OUTPUT_DIR = PY_EXAMPLE_MODEL_DIR
DEFAULT_MIRROR_DIRS = [
    PY_EXAMPLES_DIR / "scenario_mnist_kmeans/model",
    PY_EXAMPLES_DIR / "scenario_mnist_nearest_centroid/model",
]

# The LeNet-5-style backbone exports a 120-dim feature vector (its C5 layer),
# unlike common.feature_processor's own DEFAULT_PCA_COMPONENTS=160, which matches
# the CIFAR-10 backbone's 160-dim output instead.
MNIST_FEATURE_DIM = 120

# Default target: keep 95% of the 120-dim embedding's variance rather than a fixed
# component count. sklearn.decomposition.PCA(n_components=<float in (0,1)>) selects
# the *minimal* number of components whose cumulative explained_variance_ratio_ is
# >= this target - so the achieved variance is typically slightly above 95%, and the
# resulting component count depends on the actual (trained-model-specific) covariance
# structure of the extracted features. It is only known after running this script;
# _train_processor() in common/feature_processor.py prints and saves both the
# selected component count and the achieved explained-variance percentage.
MNIST_PCA_EXPLAINED_VARIANCE_TARGET = 0.95

# Import common module
sys.path.insert(0, str(ROOT_DIR / "examples/model_training"))
# Also ensure repo root for local package imports in common module
sys.path.insert(0, str(ROOT_DIR))
from common.feature_processor import (
    build_and_save_processor,
    DEFAULT_MAX_SAMPLES,
)


def _parse_pca_components(value: str) -> int | float:
    """
    --pca-components accepts either:
    - a fixed component count, e.g. "120" -> int 120
    - an explained-variance target strictly between 0 and 1, e.g. "0.95" -> float 0.95

    This matches what build_and_save_processor's pca_components parameter (and
    _train_processor underneath it) understands: a float in (0, 1) is treated as a
    variance target, anything else as a fixed component count. It matters here
    because server/benchmark/benchmark_pipeline.py's "explained_variance" PCA mode
    invokes this script with str(0.95) - plain argparse type=int cannot parse that.
    """
    f = float(value)
    if 0.0 < f < 1.0:
        return f
    if f > 0 and f == int(f):
        return int(f)
    raise argparse.ArgumentTypeError(
        f"--pca-components must be a positive whole number of components or a "
        f"variance target strictly between 0 and 1, got {value!r}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train feature processor (scaler+PCA) for MNIST")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--exclude-classes", nargs="*", default=None, help="Classes to exclude")
    p.add_argument("--img-size", type=int, default=28)
    p.add_argument(
        "--pca-components",
        type=_parse_pca_components,
        default=MNIST_PCA_EXPLAINED_VARIANCE_TARGET,
        help="Fixed component count (e.g. 120) or explained-variance target in "
             "(0, 1) (default: 0.95 -> keep 95%% of variance).",
    )
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--mirror-dir", type=Path, action="append", default=list(DEFAULT_MIRROR_DIRS),
                   help="Mirror output here (can repeat). Defaults to the "
                        "KMeans/NearestCentroid scenarios; use --no-mirror to disable.")
    p.add_argument("--no-mirror", action="store_true",
                   help="Do not mirror the processor into the other scenario folders.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Hint: run train_core.py (which writes to the Python example folder) or pass --model-path explicitly.")
        raise SystemExit(1)

    mirror_dirs = [] if args.no_mirror else args.mirror_dir

    processor_path = build_and_save_processor(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        exclude_classes=args.exclude_classes,
        img_size=args.img_size,
        pca_components=args.pca_components,
        max_samples=args.max_samples,
        mirror_dirs=mirror_dirs,
        preprocessing="lowres",
    )

    print(f"\n✓ Feature processor ready: {processor_path}")
