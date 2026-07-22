#!/usr/bin/env python3
"""
Shared training utilities for image classification scenarios.

Provides:
- build_datasets:
    Old MobileNetV2 dataset pipeline with MobileNetV2 preprocessing.
- create_model:
    Old ImageNet-pretrained MobileNetV2 transfer-learning model.
- train_and_export:
    Old MobileNetV2 training + fine-tuning + INT8 TFLite export.

Additional CIFAR10 / Low-res MobileNet utilities:
- build_datasets_cifar10_lowres:
    CIFAR10/low-res dataset pipeline with /255.0 normalization.
- create_cifar10_lowres_mobilenet_model:
    Low-res MobileNet-like model optimized for 32x32 inputs.
- train_and_export_cifar10_lowres:
    Training from scratch + INT8 TFLite export for the Low-res model.

Additional MNIST / LeNet-5-style utilities:
- build_datasets_mnist:
    MNIST dataset pipeline with /255.0 normalization, 28x28 by default.
- create_mnist_lenet5_model:
    LeNet-5-style model (LeCun et al. 1998) optimized for 28x28 inputs.
- train_and_export_mnist:
    Training from scratch + INT8 TFLite export for the LeNet-5-style model.

Important:
The old MobileNetV2 functions are intentionally kept unchanged so existing
scenarios can still use them.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import random
import shutil

import tensorflow as tf


# =============================================================================
# Old / existing MobileNetV2 defaults
# =============================================================================

DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32  # teachablemachine = 16, Grodmeier = 32
DEFAULT_EPOCHS = 15      # teachablemachine = 50, Grodmeier = 15
DEFAULT_FINE_TUNE_EPOCHS = 10
DEFAULT_LR = 0.001
DEFAULT_FINE_TUNE_LR = 0.0001


# =============================================================================
# CIFAR10 / Low-res MobileNet defaults
# =============================================================================

CIFAR10_IMG_SIZE = 32
CIFAR10_BATCH_SIZE = 64
CIFAR10_EPOCHS = 200
CIFAR10_LR = 0.1


# =============================================================================
# MNIST / LeNet-5-style defaults
# =============================================================================

MNIST_IMG_SIZE = 28
MNIST_BATCH_SIZE = 128
MNIST_EPOCHS = 15
MNIST_LR = 0.001


# =============================================================================
# Old dataset builder for ImageNet-pretrained MobileNetV2
# =============================================================================

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
    """
    Create train/val datasets from directory with optional class filtering.

    This is the old pipeline for the ImageNet-pretrained MobileNetV2 model.

    Important:
    - Uses tf.keras.applications.mobilenet_v2.preprocess_input.
    - Pixel range becomes [-1, 1].
    - Keep this for your old model.
    """
    data_dir = Path(data_dir)

    all_classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    if include_classes is not None:
        include_classes = list(include_classes)
    else:
        exclude_set = set(exclude_classes or [])
        include_classes = [c for c in all_classes if c not in exclude_set]

    if not include_classes:
        raise RuntimeError("No classes to include for training.")

    for c in all_classes:
        if c in include_classes:
            print(f"  Including class: {c}")
        else:
            print(f"  EXCLUDING class: {c} (filtered)")

    rng = random.Random(seed)

    train_files: list[str] = []
    train_labels: list[int] = []
    val_files: list[str] = []
    val_labels: list[int] = []

    # Match image extensions case-insensitively. pathlib.glob is case-sensitive
    # even on case-insensitive filesystems, so a "*.jpeg" pattern would miss
    # Imagenette's uppercase ".JPEG" files. Comparing suffix.lower() avoids that
    # and stays backward compatible with the lowercase CIFAR files.
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_idx, class_name in enumerate(include_classes):
        class_dir = data_dir / class_name
        files = sorted(
            str(p) for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

        rng.shuffle(files)
        n = len(files)

        if n == 0:
            print(f"  Warning: no images found in {class_dir}")
            continue

        split_idx = int(round(n * (1.0 - validation_split)))

        train_subset = files[:split_idx]
        val_subset = files[split_idx:]

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
        .shuffle(
            buffer_size=max(1024, len(train_files)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .map(decode_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    )

    val_ds = val_ds.map(decode_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ],
        name="old_mobilenetv2_augmentation",
    )

    train_ds = train_ds.map(
        lambda x, y: (augmentation(preprocess(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, list(include_classes)


# =============================================================================
# Old ImageNet-pretrained MobileNetV2 model
# =============================================================================

def create_model(num_classes: int, img_size: int = DEFAULT_IMG_SIZE):
    """
    Create old ImageNet-pretrained MobileNetV2 transfer-learning model.

    This is intentionally kept unchanged for your old scenarios.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")

    x = base_model(inputs, training=False)

    features = tf.keras.layers.GlobalAveragePooling2D(name="feature_pool")(x)

    x = tf.keras.layers.Dense(128, activation="relu", name="classifier_dense_128")(
        features
    )
    x = tf.keras.layers.Dropout(0.2, name="classifier_dropout")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="old_mobilenetv2_classifier")
    feature_model = tf.keras.Model(inputs, features, name="old_mobilenetv2_features")

    return model, feature_model, base_model


# =============================================================================
# Low-res MobileNet helper layers
# =============================================================================

def _relu6(x, name: str):
    """ReLU6 activation used in MobileNet-style blocks."""
    return tf.keras.layers.ReLU(max_value=6.0, name=name)(x)


def _conv_bn_relu6(
    x,
    filters: int,
    kernel_size,
    strides: int = 1,
    name: str = "conv",
):
    """
    Conv2D + BatchNorm + ReLU6 helper.

    Used for:
    - initial 3x3 conv
    - final 1x1 conv
    - optional interlayer fusion
    """
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)

    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = _relu6(x, name=f"{name}_relu6")

    return x


def _depthwise_inception_lowres(x, stride: int, name: str):
    """
    Low-res MobileNet Dwise/Inception-style block.

    stride == 1:
        Uses asymmetric depthwise kernels:
        - 3x1
        - 1x3
        - 2x2
        Then concatenates them.

    stride == 2:
        Uses:
        - 3x3 depthwise convolution with stride 2
        - 2x2 max pooling with stride 2
        Then concatenates them.

    This follows the idea from the paper:
    fill the cheap Dwise part with richer multi-scale information.
    """
    if stride == 1:
        b1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 1),
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name}_dw_3x1",
        )(x)

        b2 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, 3),
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name}_dw_1x3",
        )(x)

        b3 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(2, 2),
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name}_dw_2x2",
        )(x)

        x = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")(
            [b1, b2, b3]
        )

    elif stride == 2:
        b1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            use_bias=False,
            name=f"{name}_dw_3x3_s2",
        )(x)

        b2 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            padding="same",
            name=f"{name}_maxpool_s2",
        )(x)

        x = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")(
            [b1, b2]
        )

    else:
        raise ValueError("stride must be 1 or 2")

    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = _relu6(x, name=f"{name}_relu6")

    return x


def _lowres_bottleneck(
    x,
    out_channels: int,
    stride: int,
    expansion: int,
    dropout_rate: float,
    name: str,
):
    """
    Low-res MobileNet bottleneck.

    Differences to standard MobileNetV2:
    - Expansion 1x1 Conv is linear:
        Conv + BN, but no ReLU6.
    - Dwise layer is replaced by the Inception-style Dwise block.
    - Projection 1x1 Conv is also linear.
    - Residual connection only when stride == 1 and input/output channels match.

    The linear expansion is important because the paper argues that ReLU6 during
    dimension expansion can discard useful low-resolution information.
    """
    shortcut = x
    in_channels = int(x.shape[-1])

    hidden_channels = in_channels * expansion

    # -------------------------------------------------------------------------
    # 1) Expansion
    # -------------------------------------------------------------------------
    # For expansion == 1, the paper-style first bottleneck has no expansion.
    # For expansion > 1, use Conv + BN WITHOUT ReLU6.
    if expansion != 1:
        x = tf.keras.layers.Conv2D(
            hidden_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name}_expand_conv",
        )(x)

        x = tf.keras.layers.BatchNormalization(name=f"{name}_expand_bn")(x)

    # -------------------------------------------------------------------------
    # 2) Dwise Inception block
    # -------------------------------------------------------------------------
    x = _depthwise_inception_lowres(
        x,
        stride=stride,
        name=f"{name}_dwise_inception",
    )

    # -------------------------------------------------------------------------
    # 3) Projection
    # -------------------------------------------------------------------------
    # Linear projection back to out_channels.
    x = tf.keras.layers.Conv2D(
        out_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_project_conv",
    )(x)

    x = tf.keras.layers.BatchNormalization(name=f"{name}_project_bn")(x)

    # Dropout is placed after BN, as described in the paper.
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)

    # Residual shortcut only if shape is compatible.
    if stride == 1 and in_channels == out_channels:
        x = tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])

    return x


def _lowres_stage(
    x,
    out_channels: int,
    repeats: int,
    first_stride: int,
    expansion: int,
    dropout_rate: float,
    interlayer_concat: bool,
    name: str,
):
    """
    Build one Low-res MobileNet stage.

    Example:
        Bottleneck2:
            out_channels=24
            repeats=2
            first_stride=2
            expansion=6

    Important fix:
    The paper uses interlayer feature reuse, but the channel width should not keep
    growing without control. Therefore, after concatenating the repeated bottleneck
    outputs, we fuse them back to out_channels using a 1x1 Conv.

    Without this compression, the model becomes too large, e.g. >0.5M params.
    """
    outputs = []

    # First bottleneck in a stage may downsample.
    x = _lowres_bottleneck(
        x,
        out_channels=out_channels,
        stride=first_stride,
        expansion=expansion,
        dropout_rate=dropout_rate,
        name=f"{name}_1",
    )
    outputs.append(x)

    # Remaining bottlenecks keep spatial resolution.
    for i in range(2, repeats + 1):
        x = _lowres_bottleneck(
            x,
            out_channels=out_channels,
            stride=1,
            expansion=expansion,
            dropout_rate=dropout_rate,
            name=f"{name}_{i}",
        )
        outputs.append(x)

    # Interlayer connection / feature reuse.
    if interlayer_concat and len(outputs) > 1:
        x = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_inter_concat")(
            outputs
        )

        # Critical compression step:
        # Bring concatenated channels back to the intended stage width.
        x = _conv_bn_relu6(
            x,
            filters=out_channels,
            kernel_size=(1, 1),
            strides=1,
            name=f"{name}_inter_fuse",
        )

    return x


# =============================================================================
# New CIFAR10 / Low-res MobileNet model
# =============================================================================

def create_cifar10_lowres_mobilenet_model(
    num_classes: int,
    img_size: int = CIFAR10_IMG_SIZE,
):
    """
    Create CIFAR10 / Low-res MobileNet model for 32x32 images.

    This model is trained from scratch.

    Architecture:
        Input: 32x32x3

        Conv 3x3:
            c=32, s=1

        Bottleneck1:
            t=1, c=16, n=1, s=1

        Bottleneck2:
            t=6, c=24, n=2, s=2

        Bottleneck3:
            t=5, c=32, n=3, s=2

        Bottleneck4:
            t=3, c=64, n=4, s=2

        Bottleneck5:
            t=1, c=96, n=3, s=1

        Final Conv 1x1:
            c=160

        feature_pool:
            GlobalAveragePooling2D -> feature vector of size 160

        predictions:
            Dense softmax for training only

    The exported TFLite model uses feature_model, therefore the output is the
    160-dimensional feature vector, not the softmax predictions.
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # This model expects already normalized input:
    # Pixel values 0..255 -> 0.0..1.0
    x = inputs

    # -------------------------------------------------------------------------
    # Initial Conv: stride 1 for 32x32 low-resolution inputs.
    # -------------------------------------------------------------------------
    conv0 = _conv_bn_relu6(
        x,
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        name="initial_3x3",
    )

    # -------------------------------------------------------------------------
    # Bottleneck1: t=1, c=16, n=1, s=1
    # -------------------------------------------------------------------------
    b1 = _lowres_bottleneck(
        conv0,
        out_channels=16,
        stride=1,
        expansion=1,
        dropout_rate=0.1,
        name="bottleneck1",
    )

    # -------------------------------------------------------------------------
    # Interlayer connection:
    # Paper idea: reuse feature information.
    #
    # Important:
    # conv0 has 32 channels, b1 has 16 channels -> concat gives 48 channels.
    # We immediately fuse back to 16 channels so Bottleneck2 starts with the
    # intended width instead of an inflated width.
    # -------------------------------------------------------------------------
    x = tf.keras.layers.Concatenate(axis=-1, name="concat_initial_b1")(
        [conv0, b1]
    )

    x = _conv_bn_relu6(
        x,
        filters=16,
        kernel_size=(1, 1),
        strides=1,
        name="concat_initial_b1_fuse",
    )

    # -------------------------------------------------------------------------
    # Bottleneck2: t=6, c=24, n=2, s=2
    # -------------------------------------------------------------------------
    x = _lowres_stage(
        x,
        out_channels=24,
        repeats=2,
        first_stride=2,
        expansion=6,
        dropout_rate=0.1,
        interlayer_concat=True,
        name="bottleneck2",
    )

    # -------------------------------------------------------------------------
    # Bottleneck3: t=5, c=32, n=3, s=2
    # -------------------------------------------------------------------------
    x = _lowres_stage(
        x,
        out_channels=32,
        repeats=3,
        first_stride=2,
        expansion=5,
        dropout_rate=0.1,
        interlayer_concat=True,
        name="bottleneck3",
    )

    # -------------------------------------------------------------------------
    # Bottleneck4: t=3, c=64, n=4, s=2
    # -------------------------------------------------------------------------
    x = _lowres_stage(
        x,
        out_channels=64,
        repeats=4,
        first_stride=2,
        expansion=3,
        dropout_rate=0.1,
        interlayer_concat=True,
        name="bottleneck4",
    )

    # -------------------------------------------------------------------------
    # Bottleneck5: t=1, c=96, n=3, s=1
    #
    # No interlayer_concat here because feature maps are already small.
    # This follows the paper idea that interlayer connection is mainly useful
    # while feature maps are still relatively large.
    # -------------------------------------------------------------------------
    x = _lowres_stage(
        x,
        out_channels=96,
        repeats=3,
        first_stride=1,
        expansion=1,
        dropout_rate=0.1,
        interlayer_concat=False,
        name="bottleneck5",
    )

    # -------------------------------------------------------------------------
    # Final 1x1 Conv: c=160
    # -------------------------------------------------------------------------
    x = _conv_bn_relu6(
        x,
        filters=160,
        kernel_size=(1, 1),
        strides=1,
        name="final_1x1",
    )

    # -------------------------------------------------------------------------
    # Feature vector for KNN / drift detection.
    # For 32x32 input, spatial size should now be 4x4.
    # GlobalAveragePooling2D gives a 160-dimensional feature vector.
    # -------------------------------------------------------------------------
    features = tf.keras.layers.GlobalAveragePooling2D(name="feature_pool")(x)

    # -------------------------------------------------------------------------
    # Classification head only used during supervised training.
    # This head is NOT exported to TFLite in train_and_export_cifar10_lowres.
    # -------------------------------------------------------------------------
    x = tf.keras.layers.Dropout(0.2, name="classifier_dropout")(features)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cifar10_lowres_mobilenet",
    )

    feature_model = tf.keras.Model(
        inputs=inputs,
        outputs=features,
        name="cifar10_lowres_mobilenet_features",
    )

    # No pretrained base model exists for this custom model.
    base_model = None

    return model, feature_model, base_model


# =============================================================================
# Old MobileNetV2 training/export
# =============================================================================

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
    """
    Train old MobileNetV2 model, fine-tune, export INT8 TFLite feature model
    and labels.

    This function is intentionally kept for the old model.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)

    model, feature_model, base_model = create_model(
        num_classes,
        img_size=img_size,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Training old MobileNetV2 with {num_classes} classes:")
    print(list(class_names))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Fine-tuning old MobileNetV2...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
    )

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

    print(f"Saved old MobileNetV2 TFLite model: {tflite_path}")

    labels_path = model_dir / "labels_object.txt"
    labels_content = "\n".join(
        [f"{i} {name}" for i, name in enumerate(class_names)]
    )
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


# =============================================================================
# CIFAR10 / Low-res dataset builder
# =============================================================================

def build_datasets_cifar10_lowres(
    data_dir: Path | str,
    *,
    include_classes: Optional[Sequence[str]] = None,
    exclude_classes: Optional[Sequence[str]] = None,
    batch_size: int = CIFAR10_BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = 42,
    img_size: int = CIFAR10_IMG_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Dataset pipeline for CIFAR10 / Low-res MobileNet.

    Differences to build_datasets(...):
    - Default img_size is 32.
    - Pixel values are normalized with /255.0.
    - No MobileNetV2 preprocess_input.
    - Pixel range becomes [0, 1].

    This must match the Raspberry Pi client preprocessing:
        image = image.astype(np.float32) / 255.0
    """
    data_dir = Path(data_dir)

    all_classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    if include_classes is not None:
        include_classes = list(include_classes)
    else:
        exclude_set = set(exclude_classes or [])
        include_classes = [c for c in all_classes if c not in exclude_set]

    if not include_classes:
        raise RuntimeError("No classes to include for training.")

    for c in all_classes:
        if c in include_classes:
            print(f"  Including class: {c}")
        else:
            print(f"  EXCLUDING class: {c} (filtered)")

    rng = random.Random(seed)

    train_files: list[str] = []
    train_labels: list[int] = []
    val_files: list[str] = []
    val_labels: list[int] = []

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

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

        train_subset = files[:split_idx]
        val_subset = files[split_idx:]

        train_files.extend(train_subset)
        train_labels.extend([class_idx] * len(train_subset))

        val_files.extend(val_subset)
        val_labels.extend([class_idx] * len(val_subset))

    if not train_files:
        raise RuntimeError(f"No training images found in {data_dir}")

    if not val_files:
        raise RuntimeError(
            f"No validation images found in {data_dir}. "
            "Check validation_split or dataset size."
        )

    num_classes = len(include_classes)

    def decode_and_prepare(path: tf.Tensor, label: tf.Tensor):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)

        # Low-res model normalization:
        # 0..255 -> 0.0..1.0
        image = image / 255.0

        label_oh = tf.one_hot(label, depth=num_classes)

        return image, label_oh

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    train_ds = (
        train_ds
        .shuffle(
            buffer_size=max(1024, len(train_files)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .map(decode_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    )

    val_ds = val_ds.map(
        decode_and_prepare,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # CIFAR-like augmentation.
    # Keep it mild because input is only 32x32.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="cifar10_lowres_augmentation",
    )

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, list(include_classes)


# =============================================================================
# CIFAR10 / Low-res training/export
# =============================================================================

def train_and_export_cifar10_lowres(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: Sequence[str],
    *,
    model_dir: Path | str,
    mirror_dirs: Optional[Iterable[Path | str]] = None,
    epochs: int = CIFAR10_EPOCHS,
    learning_rate: float = CIFAR10_LR,
    img_size: int = CIFAR10_IMG_SIZE,
) -> Tuple[str, List[str]]:
    """
    Train CIFAR10 / Low-res MobileNet from scratch.

    Important:
    - No ImageNet pretraining.
    - No fine-tuning phase.
    - Uses SGD + cosine learning-rate decay.
    - Exports feature_model as INT8 TFLite.
    - TFLite output is the 160-dimensional feature vector from feature_pool.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)

    model, feature_model, _ = create_cifar10_lowres_mobilenet_model(
        num_classes=num_classes,
        img_size=img_size,
    )

    try:
        steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    except Exception:
        steps_per_epoch = 1000

    if steps_per_epoch <= 0:
        steps_per_epoch = 1000

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch * epochs,
        alpha=0.0,
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.9,
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Training CIFAR10 Low-res MobileNet with {num_classes} classes:")
    print(list(class_names))
    print(f"Image size: {img_size}x{img_size}")
    print(f"Epochs: {epochs}")
    print(f"Initial LR: {learning_rate}")
    print(f"Steps per epoch: {steps_per_epoch}")

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    def representative_dataset():
        """
        Representative dataset for INT8 quantization.

        The images here are already normalized to [0, 1], because they come from
        build_datasets_cifar10_lowres(...). This must match the model input.
        """
        for images, _ in train_ds.take(100):
            for image in images:
                yield [tf.expand_dims(image, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = model_dir / "model_object.tflite"
    tflite_path.write_bytes(tflite_model)

    print(f"Saved CIFAR10 Low-res TFLite feature model: {tflite_path}")

    labels_path = model_dir / "labels_object.txt"
    labels_content = "\n".join(
        [f"{i} {name}" for i, name in enumerate(class_names)]
    )
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


# =============================================================================
# MNIST / LeNet-5-style dataset builder
# =============================================================================

def build_datasets_mnist(
    data_dir: Path | str,
    *,
    include_classes: Optional[Sequence[str]] = None,
    exclude_classes: Optional[Sequence[str]] = None,
    batch_size: int = MNIST_BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = 42,
    img_size: int = MNIST_IMG_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Dataset pipeline for the MNIST LeNet-5-style backbone.

    Images are exported as single-channel grayscale PNGs (see
    examples/dataset_mnist/export_mnist.py), but decoded here with channels=3:
    every other consumer of these images (create_initial_state.py's cv2.imread,
    main.py's runtime FolderImageSource) also always produces 3 channels, since
    cv2.imread replicates grayscale into 3 identical channels by default. Decoding
    as channels=3 here keeps training consistent with that instead of introducing a
    separate single-channel code path the shared runtime does not support.

    Pixel values are normalized with /255.0, matching the Raspberry Pi client:
        image = image.astype(np.float32) / 255.0
    """
    data_dir = Path(data_dir)

    all_classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    if include_classes is not None:
        include_classes = list(include_classes)
    else:
        exclude_set = set(exclude_classes or [])
        include_classes = [c for c in all_classes if c not in exclude_set]

    if not include_classes:
        raise RuntimeError("No classes to include for training.")

    for c in all_classes:
        if c in include_classes:
            print(f"  Including class: {c}")
        else:
            print(f"  EXCLUDING class: {c} (filtered)")

    rng = random.Random(seed)

    train_files: list[str] = []
    train_labels: list[int] = []
    val_files: list[str] = []
    val_labels: list[int] = []

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

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

        train_subset = files[:split_idx]
        val_subset = files[split_idx:]

        train_files.extend(train_subset)
        train_labels.extend([class_idx] * len(train_subset))

        val_files.extend(val_subset)
        val_labels.extend([class_idx] * len(val_subset))

    if not train_files:
        raise RuntimeError(f"No training images found in {data_dir}")

    if not val_files:
        raise RuntimeError(
            f"No validation images found in {data_dir}. "
            "Check validation_split or dataset size."
        )

    num_classes = len(include_classes)

    def decode_and_prepare(path: tf.Tensor, label: tf.Tensor):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)

        # LeNet-5-style model normalization:
        # 0..255 -> 0.0..1.0
        image = image / 255.0

        label_oh = tf.one_hot(label, depth=num_classes)

        return image, label_oh

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    train_ds = (
        train_ds
        .shuffle(
            buffer_size=max(1024, len(train_files)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .map(decode_and_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    )

    val_ds = val_ds.map(
        decode_and_prepare,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Mild augmentation only. No horizontal/vertical flip: flipping would change a
    # digit's meaning (e.g. some digits become confusable with others), unlike the
    # CIFAR-10 natural-image augmentation.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="mnist_lenet5_augmentation",
    )

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, list(include_classes)


# =============================================================================
# MNIST / LeNet-5-style model
# =============================================================================

def create_mnist_lenet5_model(
    num_classes: int,
    img_size: int = MNIST_IMG_SIZE,
):
    """
    Create a LeNet-5-style model for MNIST-like 28x28 inputs.

    Architecture follows LeCun, Bottou, Bengio & Haffner, "Gradient-Based Learning
    Applied to Document Recognition", Proc. IEEE 86(11), 1998 (LeNet-5). The original
    pads its 28x28 input to 32x32 and uses "valid" convolutions throughout; using
    "same" padding on C1 with the 28x28 input directly produces the identical spatial
    trajectory (28x28x6 -> 14x14x6 -> 10x10x16 -> 5x5x16) without an explicit padding
    step. S2/S4 use **average** pooling (the original's subsampling layers), not max
    pooling.

    Two deliberate adaptations to this framework's conventions:
    - BatchNorm + ReLU6 after each conv instead of the original tanh/sigmoid
      activations, matching the Low-res MobileNet's INT8-export-friendly activation
      choice above.
    - C5 is implemented as Flatten + Dense(120) rather than a 5x5 "fully-connected
      convolution" — mathematically equivalent given the 5x5x16 input, since a Conv2D
      whose kernel exactly matches the input's spatial size collapses to a dense
      layer over the flattened receptive field.

    See server/Retrainer/README.md -> "Feature Extractor Architectures Per Dataset"
    for the citation and rationale.

    Input is 3-channel (see build_datasets_mnist for why), even though the source
    images are grayscale.

        Input: 28x28x3

        C1 - Conv 5x5, 6 filters, stride 1, "same":
            -> 28x28x6
        S2 - Average pooling 2x2:
            -> 14x14x6

        C3 - Conv 5x5, 16 filters, stride 1, "valid":
            -> 10x10x16
        S4 - Average pooling 2x2:
            -> 5x5x16

        Flatten -> C5 - Dense 120:
            feature_vector, exported (120-dim, matches the original LeNet-5 C5 layer)

        F6 - Dense 84 + predictions - Dense(num_classes, softmax):
            training-only classification head, not exported.
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")

    x = tf.keras.layers.Conv2D(
        6, 5, strides=1, padding="same", use_bias=False, name="c1_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="c1_bn")(x)
    x = _relu6(x, name="c1_relu6")
    x = tf.keras.layers.AveragePooling2D(2, name="s2_pool")(x)

    x = tf.keras.layers.Conv2D(
        16, 5, strides=1, padding="valid", use_bias=False, name="c3_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="c3_bn")(x)
    x = _relu6(x, name="c3_relu6")
    x = tf.keras.layers.AveragePooling2D(2, name="s4_pool")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)

    features = tf.keras.layers.Dense(120, activation="relu", name="c5_dense_120")(x)

    x = tf.keras.layers.Dense(84, activation="relu", name="f6_dense_84")(features)
    x = tf.keras.layers.Dropout(0.2, name="classifier_dropout")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_lenet5")
    feature_model = tf.keras.Model(
        inputs=inputs, outputs=features, name="mnist_lenet5_features"
    )

    # No pretrained base model exists for this custom model.
    base_model = None

    return model, feature_model, base_model


# =============================================================================
# MNIST / LeNet-5-style training/export
# =============================================================================

def train_and_export_mnist(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: Sequence[str],
    *,
    model_dir: Path | str,
    mirror_dirs: Optional[Iterable[Path | str]] = None,
    epochs: int = MNIST_EPOCHS,
    learning_rate: float = MNIST_LR,
    img_size: int = MNIST_IMG_SIZE,
    early_stopping_patience: int = 3,
) -> Tuple[str, List[str]]:
    """
    Train the LeNet-5-style MNIST model from scratch and export INT8 TFLite.

    Simpler optimizer setup than the CIFAR-10 Low-res MobileNet (Adam with a fixed
    learning rate, no cosine schedule/momentum) because this much smaller model on
    MNIST converges quickly and reliably without the additional tuning the deeper
    CIFAR-10 backbone needed.

    MNIST saturates this model very quickly: val_accuracy plateaus within the first
    3-4 epochs and val_loss stops improving shortly after, while train accuracy keeps
    climbing - i.e. every epoch beyond that point is pure overfitting, not learning.
    EarlyStopping(monitor="val_loss", restore_best_weights=True) both stops training
    once val_loss stops improving for `early_stopping_patience` epochs, and - just as
    important - restores the model's weights to the best (lowest val_loss) epoch
    before export. Without restore_best_weights, the exported model corresponds to
    whichever epoch training happened to end on, which can be worse than an earlier
    epoch (val_loss is noisy near the plateau, not monotonically improving). `epochs`
    remains an upper bound; in practice training usually stops well before it.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)

    model, feature_model, _ = create_mnist_lenet5_model(
        num_classes=num_classes,
        img_size=img_size,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Training MNIST LeNet-5-style model with {num_classes} classes:")
    print(list(class_names))
    print(f"Image size: {img_size}x{img_size}")
    print(f"Epochs (upper bound): {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience (val_loss): {early_stopping_patience}")

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    def representative_dataset():
        """
        Representative dataset for INT8 quantization.

        The images here are already normalized to [0, 1], because they come from
        build_datasets_mnist(...). This must match the model input.
        """
        for images, _ in train_ds.take(100):
            for image in images:
                yield [tf.expand_dims(image, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = model_dir / "model_object.tflite"
    tflite_path.write_bytes(tflite_model)

    print(f"Saved MNIST LeNet-5-style TFLite feature model: {tflite_path}")

    labels_path = model_dir / "labels_object.txt"
    labels_content = "\n".join(
        [f"{i} {name}" for i, name in enumerate(class_names)]
    )
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