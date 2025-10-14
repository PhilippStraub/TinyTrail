#!/usr/bin/env python3
"""
Shared feature processor (StandardScaler + PCA) training utilities.

Functions:
- build_and_save_processor: extracts features via TFLite model, trains scaler+PCA,
  saves as .npz, optional mirroring to other directories.
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


DEFAULT_IMG_SIZE = 224
DEFAULT_PCA_COMPONENTS = 256
DEFAULT_MAX_SAMPLES = 2000


def _load_tflite_model(model_path: Path | str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _extract_features(
    interpreter: tf.lite.Interpreter,
    image_path: Path,
    *,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocessing to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_dtype = input_details.get('dtype')
    in_scale, in_zero = get_qparams(input_details)

    if in_dtype in (np.int8, np.uint8):
        qin = np.round(img_array / max(in_scale, 1e-12) + in_zero)
        if in_dtype == np.int8:
            qin = np.clip(qin, -128, 127).astype(np.int8)
        else:
            qin = np.clip(qin, 0, 255).astype(np.uint8)
        interpreter.set_tensor(input_details['index'], qin)
    else:
        interpreter.set_tensor(input_details['index'], img_array.astype(in_dtype))

    interpreter.invoke()

    out = interpreter.get_tensor(output_details['index'])
    out_dtype = output_details.get('dtype')
    out_scale, out_zero = get_qparams(output_details)

    if out_dtype in (np.int8, np.uint8):
        features = (out.astype(np.float32) - out_zero) * out_scale
    else:
        features = out.astype(np.float32)

    return features.flatten()


def _collect_features(
    model_path: Path | str,
    data_dir: Path | str,
    *,
    exclude_classes: Optional[Sequence[str]] = None,
    img_size: int = DEFAULT_IMG_SIZE,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> Tuple[np.ndarray, list[str]]:
    interpreter = _load_tflite_model(model_path)
    data_dir = Path(data_dir)
    exclude = set(exclude_classes or [])

    features_list: list[np.ndarray] = []
    labels_list: list[str] = []

    classes = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    for class_dir in classes:
        cname = class_dir.name
        if cname in exclude:
            print(f"EXCLUDING {cname} (filtered)")
            continue

        print(f"Processing {cname}…")
        imgs = list(class_dir.glob("*.jpg"))
        # Basic balancing across classes
        limit = max(1, max_samples // max(1, len(classes)))
        for img_path in imgs[:limit]:
            try:
                feats = _extract_features(interpreter, img_path, img_size=img_size)
                features_list.append(feats)
                labels_list.append(cname)
            except Exception as e:
                print(f"  Warning: Failed to process {img_path}: {e}")

    features_array = np.array(features_list, dtype=np.float32)
    print(f"Collected {len(features_array)} feature vectors")
    return features_array, labels_list


def _train_processor(
    features: np.ndarray,
    *,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
) -> dict:
    print(f"Training StandardScaler on {features.shape}")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    print(f"Training PCA: {features.shape[1]}D → {pca_components}D")
    pca = PCA(n_components=pca_components, random_state=42)
    pca.fit(scaled)

    explained = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    print(f"PCA explains {explained:.1f}% of variance")

    return {
        'mu': scaler.mean_.astype(np.float32),
        'inv_scale': (1.0 / scaler.scale_).astype(np.float32),
        'W': pca.components_.astype(np.float32),
        'input_dim': int(features.shape[1]),
        'output_dim': int(pca_components),
        'explained_variance': explained,
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
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, _ = _collect_features(
        model_path, data_dir,
        exclude_classes=exclude_classes,
        img_size=img_size,
        max_samples=max_samples,
    )

    processor = _train_processor(features, pca_components=pca_components)

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
