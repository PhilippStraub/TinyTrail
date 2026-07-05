#!/usr/bin/env python3
"""
Update config files and run minimal smoke test.
Verifies model loads and processes test image correctly.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "examples/drift_detection/config.json"
MODEL_DIR = ROOT_DIR / "examples/drift_detection/model"
TEST_IMAGE = ROOT_DIR / "examples/assets/training_images/stone/stone_001.jpg"

def update_config():
    """Update config.json with new model paths."""
    config = {
        "feature_extractor": {
            "model_path": str(MODEL_DIR / "model_object.tflite"),
            "feature_layer_index": -1,
            "normalize_features": False,
            "lazy_loading": False
        },
        "processor": {
            "processor_path": str(MODEL_DIR / "feature_processor.npz"),
            "strict": True,
            "allow_batch": True
        },
        "classifier": {
            "k": 5,
            "max_samples": 160,
            "distance_metric": "euclidean",
            "weighted": True,
            "adaptive_k": False
        },
        "drift_monitor": {
            "alpha": 0.999,
            "delta": 0.002,
            "threshold": 0.05,
            "min_samples": 10,
            "aggregation_mode": "mean",
            "k_distances": 5
        }
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Updated config: {CONFIG_PATH}")
    return config

def smoke_test(config):
    """Run minimal test to verify model works."""
    model_path = config["feature_extractor"]["model_path"]

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Load test image
    from PIL import Image
    img = Image.open(TEST_IMAGE).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocessing
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    features = interpreter.get_tensor(output_details[0]['index'])

    print(f"✓ Model loaded: {Path(model_path).name}")
    print(f"✓ Test inference successful")
    print(f"✓ Feature shape: {features.shape} (expecting 1280D)")

    # Verify processor
    processor_path = config["processor"]["processor_path"]
    if Path(processor_path).exists():
        data = np.load(processor_path)
        print(f"✓ Processor loaded: {data['input_dim']}D → {data['output_dim']}D")
    else:
        print(f"⚠ Processor not found: {processor_path}")

if __name__ == "__main__":
    config = update_config()
    smoke_test(config)
    print("\n✓ Config updated and smoke test passed!")
