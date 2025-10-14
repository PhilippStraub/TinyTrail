#!/usr/bin/env python3
"""
Shared training utilities for image classification scenarios.

Provides:
- build_datasets: create train/val tf.data pipelines with optional class filtering
- create_model: MobileNetV2 head with GAP feature vector
- train_and_export: train, fine-tune, export TFLite INT8 + labels, optional mirroring
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import random
import shutil

import tensorflow as tf


# Default hyperparameters
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32 #teachablemachine = 16 #Grodmeier = 32
DEFAULT_EPOCHS = 15 #teachablemachine = 50 #Grodmeier = 15
DEFAULT_FINE_TUNE_EPOCHS = 10
DEFAULT_LR = 0.001
DEFAULT_FINE_TUNE_LR = 0.0001


def build_datasets(
    data_dir: Path | str,
    *,
    include_classes: Optional[Sequence[str]] = None,
    exclude_classes: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = 42,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Create train/val datasets from directory with optional class filtering.

    - If `include_classes` is provided, uses it verbatim (must exist under data_dir).
    - Else, derives classes from subdirectories and excludes any in `exclude_classes`.
    """
    data_dir = Path(data_dir)

    # Determine included classes
    all_classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if include_classes is not None:
        include_classes = list(include_classes)
    else:
        exclude_set = set(exclude_classes or [])
        include_classes = [c for c in all_classes if c not in exclude_set]

    if not include_classes:
        raise RuntimeError("No classes to include for training.")

    # Log inclusion/exclusion
    for c in all_classes:
        if c in include_classes:
            print(f"  Including class: {c}")
        else:
            print(f"  EXCLUDING class: {c} (filtered)")

    # Build stratified file lists per class
    rng = random.Random(seed)
    train_files: list[str] = []
    train_labels: list[int] = []
    val_files: list[str] = []
    val_labels: list[int] = []

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for class_idx, class_name in enumerate(include_classes):
        class_dir = data_dir / class_name
        files: list[str] = []
        for pat in patterns:
            files.extend(sorted(str(p) for p in class_dir.glob(pat)))

        rng.shuffle(files)
        n = len(files)
        if n == 0:
            print(f"  Warning: no images found in {class_dir}")
            continue

        split_idx = int(round(n * (1.0 - validation_split)))
        train_subset, val_subset = files[:split_idx], files[split_idx:]

        train_files.extend(train_subset)
        train_labels.extend([class_idx] * len(train_subset))
        val_files.extend(val_subset)
        val_labels.extend([class_idx] * len(val_subset))

    num_classes = len(include_classes)

    def decode_and_prepare(path: tf.Tensor, label: tf.Tensor):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        label_oh = tf.one_hot(label, depth=num_classes)
        return image, label_oh

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    train_ds = (
        train_ds
        .shuffle(buffer_size=max(1024, len(train_files)), seed=seed, reshuffle_each_iteration=True)
        .map(decode_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    )
    val_ds = val_ds.map(decode_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    train_ds = train_ds.map(lambda x, y: (augmentation(preprocess(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, list(include_classes)


def create_model(num_classes: int, img_size: int = DEFAULT_IMG_SIZE):
    """Create MobileNetV2 transfer learning model and feature extractor."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name='feature_pool')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(features)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    feature_model = tf.keras.Model(inputs, features)
    return model, feature_model, base_model


def train_and_export(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: Sequence[str],
    *,
    model_dir: Path | str,
    mirror_dirs: Optional[Iterable[Path | str]] = None,
    epochs: int = DEFAULT_EPOCHS,
    fine_tune_epochs: int = DEFAULT_FINE_TUNE_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    fine_tune_lr: float = DEFAULT_FINE_TUNE_LR,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Tuple[str, List[str]]:
    """Train model, fine-tune, export to TFLite INT8 + labels, optionally mirror files."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    model, feature_model, base_model = create_model(num_classes, img_size=img_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Training with {num_classes} classes: {list(class_names)}")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Fine-tuning...")
    model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs)

    def representative_dataset():
        for images, _ in train_ds.take(100):
            for image in images:
                yield [tf.expand_dims(image, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = model_dir / "model_object.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Saved TFLite model: {tflite_path}")

    labels_path = model_dir / "labels_object.txt"
    labels_content = '\n'.join([f"{i} {name}" for i, name in enumerate(class_names)])
    labels_path.write_text(labels_content)
    print(f"Saved labels: {labels_path}")

    if mirror_dirs:
        for md in mirror_dirs:
            md = Path(md)
            md.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tflite_path, md / tflite_path.name)
            shutil.copy2(labels_path, md / labels_path.name)
            print(f"Mirrored artifacts to: {md}")

    return str(tflite_path), list(class_names)

