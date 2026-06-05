#!/usr/bin/env python3
"""
Shared feature processor (StandardScaler + PCA) training utilities.

Functions:
- build_and_save_processor:
    Extracts features via TFLite model, trains StandardScaler + PCA,
    saves as .npz, optional mirroring to other directories.

Important for CIFAR10 / Low-res MobileNet:
- Supports PNG images.
- Uses low-res preprocessing by default:
    pixel values 0..255 -> 0.0..1.0
- This must match the Raspberry Pi client preprocessing.
"""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import sys
import shutil

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image

try:
    from tinylcm_python.qparams import get_qparams
except ImportError:
    # Ensure repo root is on sys.path for local package imports
    ROOT_DIR = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(ROOT_DIR))
    from tinylcm_python.qparams import get_qparams


# Old default was 224 for ImageNet MobileNetV2.
# Your CIFAR10 wrapper passes img_size=32, so this default is mostly fallback.
DEFAULT_IMG_SIZE = 224

# Your Low-res MobileNet feature vector has 160 dimensions.
# If PCA components are higher than feature dimension, we automatically reduce them.
DEFAULT_PCA_COMPONENTS = 160

# Max number of images used for processor training.
DEFAULT_MAX_SAMPLES = 2000

# Supported image file extensions.
SUPPORTED_IMAGE_PATTERNS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp",
)


def _load_tflite_model(model_path: Path | str) -> tf.lite.Interpreter:
    """
    Load a TensorFlow Lite model and allocate tensors.
    """
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _preprocess_image_lowres(
    image_path: Path,
    *,
    img_size: int,
) -> np.ndarray:
    """
    Preprocessing for CIFAR10 / Low-res MobileNet.

    Steps:
    1. Load image as RGB.
    2. Resize to img_size x img_size, usually 32x32.
    3. Convert to float32.
    4. Normalize pixel values from 0..255 to 0.0..1.0.

    This must match your Raspberry Pi client:
        image = image.astype(np.float32) / 255.0
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0

    # Add batch dimension:
    # (32, 32, 3) -> (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def _preprocess_image_mobilenetv2(
    image_path: Path,
    *,
    img_size: int,
) -> np.ndarray:
    """
    Old preprocessing for ImageNet MobileNetV2.

    Pixel range:
        0..255 -> -1.0..1.0

    Keep this only for old MobileNetV2 models.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))

    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def _extract_features(
    interpreter: tf.lite.Interpreter,
    image_path: Path,
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    preprocessing: str = "lowres",
) -> np.ndarray:
    """
    Extract one feature vector from one image using the TFLite model.

    preprocessing:
        "lowres":
            For your new CIFAR10 / Low-res MobileNet.
            Uses /255.0.

        "mobilenetv2":
            For the old ImageNet MobileNetV2 model.
            Uses (x / 127.5) - 1.0.
    """
    if preprocessing == "lowres":
        img_array = _preprocess_image_lowres(image_path, img_size=img_size)
    elif preprocessing == "mobilenetv2":
        img_array = _preprocess_image_mobilenetv2(image_path, img_size=img_size)
    else:
        raise ValueError(
            f"Unknown preprocessing mode: {preprocessing}. "
            "Use 'lowres' or 'mobilenetv2'."
        )

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_dtype = input_details.get("dtype")
    in_scale, in_zero = get_qparams(input_details)

    # -------------------------------------------------------------------------
    # Input quantization
    # -------------------------------------------------------------------------
    # Your exported model is INT8. The model was calibrated with representative
    # data in the same preprocessing range. For lowres this is [0, 1].
    if in_dtype in (np.int8, np.uint8):
        qin = np.round(img_array / max(in_scale, 1e-12) + in_zero)

        if in_dtype == np.int8:
            qin = np.clip(qin, -128, 127).astype(np.int8)
        else:
            qin = np.clip(qin, 0, 255).astype(np.uint8)

        interpreter.set_tensor(input_details["index"], qin)
    else:
        interpreter.set_tensor(input_details["index"], img_array.astype(in_dtype))

    interpreter.invoke()

    out = interpreter.get_tensor(output_details["index"])

    out_dtype = output_details.get("dtype")
    out_scale, out_zero = get_qparams(output_details)

    # -------------------------------------------------------------------------
    # Output dequantization
    # -------------------------------------------------------------------------
    # Convert INT8/UINT8 feature output back to float32 feature vector.
    if out_dtype in (np.int8, np.uint8):
        features = (out.astype(np.float32) - out_zero) * out_scale
    else:
        features = out.astype(np.float32)

    return features.flatten()


def _find_images(class_dir: Path) -> list[Path]:
    """
    Collect image files from one class directory.

    Important:
    Your CIFAR10 dataset uses PNG files, therefore '*.png' must be included.
    """
    image_files: list[Path] = []

    for pattern in SUPPORTED_IMAGE_PATTERNS:
        image_files.extend(sorted(class_dir.glob(pattern)))

    return image_files


def _collect_features(
    model_path: Path | str,
    data_dir: Path | str,
    *,
    exclude_classes: Optional[Sequence[str]] = None,
    img_size: int = DEFAULT_IMG_SIZE,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    preprocessing: str = "lowres",
) -> Tuple[np.ndarray, list[str]]:
    """
    Extract feature vectors for images in data_dir.

    Expected directory structure:
        data_dir/
            airplane/
                image1.png
                image2.png
            automobile/
                image1.png
                ...

    Returns:
        features_array:
            Shape: (n_samples, feature_dim)
            For your Low-res model feature_dim should be 160.

        labels_list:
            Class label for each feature vector.
    """
    interpreter = _load_tflite_model(model_path)

    data_dir = Path(data_dir)
    exclude = set(exclude_classes or [])

    features_list: list[np.ndarray] = []
    labels_list: list[str] = []

    all_classes = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    included_classes = [d for d in all_classes if d.name not in exclude]

    if not included_classes:
        raise RuntimeError(
            f"No class folders found in {data_dir} after applying exclude_classes."
        )

    # Basic balancing across included classes.
    limit_per_class = max(1, max_samples // max(1, len(included_classes)))

    for class_dir in all_classes:
        cname = class_dir.name

        if cname in exclude:
            print(f"EXCLUDING {cname} (filtered)")
            continue

        print(f"Processing {cname}…")

        imgs = _find_images(class_dir)

        print(f"  Found {len(imgs)} image files")

        if not imgs:
            print(f"  Warning: no supported images found in {class_dir}")
            continue

        for img_path in imgs[:limit_per_class]:
            try:
                feats = _extract_features(
                    interpreter,
                    img_path,
                    img_size=img_size,
                    preprocessing=preprocessing,
                )

                features_list.append(feats)
                labels_list.append(cname)

            except Exception as e:
                print(f"  Warning: Failed to process {img_path}: {e}")

    if not features_list:
        raise RuntimeError(
            "Collected 0 feature vectors. "
            "Check image file extensions, data_dir, model_path and preprocessing mode."
        )

    features_array = np.array(features_list, dtype=np.float32)

    print(f"Collected {len(features_array)} feature vectors")
    print(f"Feature shape: {features_array.shape}")

    return features_array, labels_list


def _train_processor(
    features: np.ndarray,
    *,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
) -> dict:
    """
    Train StandardScaler and PCA on extracted features.

    Important:
    PCA components cannot be larger than:
        min(n_samples, feature_dim)

    Your Low-res MobileNet feature vector usually has 160 dimensions.
    Therefore pca_components=256 would be invalid.
    This function automatically reduces it.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected features to be a 2D array, got shape {features.shape}"
        )

    n_samples, input_dim = features.shape

    if n_samples == 0:
        raise ValueError("No feature vectors available for processor training.")

    max_valid_components = min(n_samples, input_dim)

    if pca_components > max_valid_components:
        print(
            f"Requested PCA components={pca_components}, "
            f"but max valid is {max_valid_components}. "
            f"Using {max_valid_components} instead."
        )
        pca_components = max_valid_components

    print(f"Training StandardScaler on {features.shape}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    print(f"Training PCA: {input_dim}D → {pca_components}D")

    pca = PCA(
        n_components=pca_components,
        random_state=42,
    )
    pca.fit(scaled)

    explained = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    print(f"PCA explains {explained:.1f}% of variance")

    return {
        "mu": scaler.mean_.astype(np.float32),
        "inv_scale": (1.0 / scaler.scale_).astype(np.float32),
        "W": pca.components_.astype(np.float32),
        "input_dim": int(input_dim),
        "output_dim": int(pca_components),
        "explained_variance": explained,
    }


def build_and_save_processor(
    *,
    model_path: Path | str,
    data_dir: Path | str,
    output_dir: Path | str,
    exclude_classes: Optional[Sequence[str]] = None,
    img_size: int = DEFAULT_IMG_SIZE,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    mirror_dirs: Optional[Iterable[Path | str]] = None,
    preprocessing: str = "lowres",
) -> Path:
    """
    Build and save feature processor.

    For your new CIFAR10 / Low-res MobileNet use:
        preprocessing="lowres"
        img_size=32

    For the old ImageNet MobileNetV2 model use:
        preprocessing="mobilenetv2"
        img_size=224 or whatever the old model expects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, _ = _collect_features(
        model_path,
        data_dir,
        exclude_classes=exclude_classes,
        img_size=img_size,
        max_samples=max_samples,
        preprocessing=preprocessing,
    )

    processor = _train_processor(
        features,
        pca_components=pca_components,
    )

    processor_path = output_dir / "feature_processor.npz"

    np.savez(processor_path, **processor)

    print(f"Saved processor: {processor_path}")

    if mirror_dirs:
        for md in mirror_dirs:
            mdp = Path(md)
            mdp.mkdir(parents=True, exist_ok=True)

            dst = mdp / processor_path.name
            shutil.copy2(processor_path, dst)

            print(f"Mirrored processor to: {dst}")

    return processor_path