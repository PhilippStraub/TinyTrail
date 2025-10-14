#!/usr/bin/env python3
"""
On-Device Drift Detection System v3 metrics-only
------------------------------------------
This system demonstrates the inference pipeline of system v3 with framework drift-detection components deactivated.
!! Must be executed directly without deployment executer, to prevent use of other client framework components.

Designed for Raspberry Pi Zero 2W.
"""

import argparse
import json
import logging
import os
import psutil
import signal
import socket
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# tflite_runtime is handled by TFLiteFeatureExtractor class

# Ensure project root is on path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utils explicitly to avoid name conflicts with other 'utils' packages
from utils.camera_handler import CameraHandler
from utils.preprocessors import resize_image

# Project root is already added above for tinylcm imports

from tinylcm_python.knn import LightweightKNN
# Baseline-nodrift
# from tinylcm_python.drift_monitor import KNNDistanceMonitor
from tinylcm_python.feature_extractor import TFLiteFeatureExtractor
from tinylcm_python.scaler_pca import StandardScalerPCATransformer

from rover import rover
# Global variables
logger = None
running = True
config = None
feature_transformer = None
process = psutil.Process(os.getpid())
speed = 25 #Rover

def crop_and_resize(img,inference_resolution):
    """Crops and resizes images to make left, center, and right image."""
    height, width, _ = img.shape
    region_width = width // 3
    crops = []
    # Accept both int and tuple for inference_resolution
    if isinstance(inference_resolution, tuple):
        target_size = (inference_resolution[0], inference_resolution[1])
    else:
        target_size = (inference_resolution, inference_resolution)
    for i in range(3):
        start_x = i * region_width
        end_x = start_x + region_width
        crop = img[:, start_x:end_x]
        h, w, _ = crop.shape
        if h > w:
            pad = (h - w) // 2
            crop = crop[pad:pad+w, :]
        elif w > h:
            pad = (w - h) // 2
            crop = crop[:, pad:pad+h]
        crop_resized = cv2.resize(crop, target_size)
        crops.append(crop_resized)
    return crops


def load_config(config_path: str) -> Dict:
    """Load the configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    global logger
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set up logging
    log_file = logs_dir / f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("drift_detection")
    logger.info(f"Logging initialized at level {log_level}")


def create_directory_structure(config: Dict) -> None:
    """Create necessary directories from configuration."""
    # Create log directory
    log_dir = Path(config["tinylcm"]["data_logger"]["log_dir"])
    log_dir.mkdir(exist_ok=True)
    
    # Create debug directory if needed
    if config["application"].get("save_debug_frames", False):
        debug_dir = Path(config["application"]["debug_output_dir"])
        debug_dir.mkdir(exist_ok=True)


class PerformanceLogger:
    """Performance logger for drift detection metrics."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize the performance logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create performance log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"performance_{timestamp}.json"
        
        # Performance metrics storage
        self.metrics = []
        self.start_time = time.time()
        
        # Log system info at start
        self.log_system_info()
    
    def log_system_info(self):
        """Log system information at startup."""
        import platform
        
        system_info = {
            "timestamp": time.time(),
            "type": "system_info",
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024)
        }
        
        self.metrics.append(system_info)
    
    def log_inference(self, prediction: str, confidence: float, latency_ms: float, 
                     drift_detected: bool = False, cpu_value: Optional[float] = None, memory_value: Optional[float] = None):
        """Log inference metrics."""
        metric = {
            "timestamp": time.time(),
            "type": "inference",
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "drift_detected": drift_detected,
            "cpu_percent": cpu_value if cpu_value is not None else psutil.cpu_percent(),
            "memory_percent": memory_value if memory_value is not None else psutil.virtual_memory().percent
        }
        
        self.metrics.append(metric)
    
    def log_drift_event(self, detector_name: str, drift_type: str, details: Dict):
        """Log drift detection event."""
        event = {
            "timestamp": time.time(),
            "type": "drift_event",
            "detector": detector_name,
            "drift_type": drift_type,
            "details": details
        }
        
        self.metrics.append(event)
        
        # Also save immediately for drift events
        self.save()
    
    def save(self):
        """Save metrics to JSON file."""
        try:
            # Cast all numpy types to native python types
            def default_serializer(obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                return str(obj)  # fallback
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=default_serializer)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics:
            return {}
        
        inference_metrics = [m for m in self.metrics if m.get("type") == "inference"]
        drift_events = [m for m in self.metrics if m.get("type") == "drift_event"]
        
        if inference_metrics:
            latencies = [m["latency_ms"] for m in inference_metrics]
            confidences = [m["confidence"] for m in inference_metrics if "confidence" in m and m["confidence"] is not None]
            
            return {
                "total_inferences": len(inference_metrics),
                "total_drift_events": len(drift_events),
                "avg_latency_ms": np.mean(latencies),
                "max_latency_ms": np.max(latencies),
                "min_latency_ms": np.min(latencies),
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "runtime_seconds": time.time() - self.start_time
            }
        
        return {}


# Global performance logger
performance_logger = None

def handle_sigterm(signum, frame):
    """Handle termination signals for graceful shutdown."""
    global running
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    running = False


def process_labels(labels_path: str) -> List[str]:
    """Process label file to extract class names."""
    with open(labels_path, 'r') as f:
        labels_raw = [line.strip() for line in f.readlines()]
    
    # Extract actual class names (format: "0 red" -> "red")
    labels = []
    for label in labels_raw:
        parts = label.split(' ', 1)
        if len(parts) > 1:
            labels.append(parts[1])
        else:
            labels.append(label)
    
    return labels


def center_square(img: np.ndarray) -> np.ndarray:
    """Extract center square crop from image to reduce background noise.
    
    Args:
        img: Input image as numpy array
    
    Returns:
        Center square crop of the image
    """
    h, w = img.shape[:2]
    m = min(h, w)
    y = (h - m) // 2
    x = (w - m) // 2
    return img[y:y+m, x:x+m]


def apply_feature_transformation(features: np.ndarray) -> np.ndarray:
    """Apply feature transformation (PCA) to extracted features."""
    global feature_transformer
    
    if feature_transformer is None:
        logger.warning("Feature transformer not initialized, returning original features")
        return features
    
    try:
        # Apply transformation
        transformed_features = feature_transformer.transform(features)
        return transformed_features
    
    except Exception as e:
        logger.error(f"Error applying feature transformation: {e}")
        return features


def main():
    """Main function for the drift detection example."""
    global running, config, feature_transformer, performance_logger
    
    parser = argparse.ArgumentParser(description="Minimal On-Device Drift Detection")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config["device"]["log_level"])
    
    # Create directory structure
    create_directory_structure(config)
    
    # Initialize performance logger
    performance_logger = PerformanceLogger()
    # Warm-up psutil CPU percent so the first measured value isn't 0.0 by definition
    try:
        _ = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info("Starting Minimal On-Device Drift Detection")
    
    try:
        # Get device ID (simple hostname-based ID for minimal version)
        device_id = socket.gethostname() or "drift-detector-001"
        logger.info(f"Device ID: {device_id}")
        
        # Initialize CameraHandler
        camera_config = config["camera"]
        camera = CameraHandler(
            resolution=tuple(camera_config["resolution"]),
            framerate=camera_config["framerate"],
            rotation=camera_config.get("rotation", 0),
            camera_index=camera_config.get("camera_index", 0),
            force_cv2=camera_config.get("force_cv2", False)
        )
        # Prime: wait for first frame to avoid early None frames
        try:
            first_frame = getattr(camera, 'wait_for_frame', lambda timeout=2.0: None)(timeout=2.0)
            if first_frame is None:
                logger.warning("No camera frame received during startup warm-up (2s)")
            else:
                logger.info(f"Received initial camera frame: shape={getattr(first_frame, 'shape', None)}")
        except Exception as e:
            logger.debug(f"Camera warm-up wait failed: {e}")
        
        # Load TFLite model labels
        model_config = config["model"]
        labels_path = model_config["labels_path"]
        labels = process_labels(labels_path)
        logger.info(f"Loaded {len(labels)} labels: {labels}")
        
        # Initialize TinyLCM components
        tinylcm_config = config["tinylcm"]
        
        feature_extractor_config = tinylcm_config["feature_extractor"]

        def preprocess_image(image):
            """Preprocess image for the TFLite model."""
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # MobileNetV2: x_norm = (x / 127.5) - 1.0 ∈ [-1,1]
            image = image.astype(np.float32)
            image = (image / 127.5) - 1.0
            return image

        feature_extractor = TFLiteFeatureExtractor(
            model_path=feature_extractor_config["model_path"],
            feature_layer_index=feature_extractor_config["feature_layer_index"],
            normalize_features=feature_extractor_config.get("normalize_features", False),
            lazy_loading=feature_extractor_config.get("lazy_loading", False),
            preprocessors=[preprocess_image]
        )

        if "feature_transformation" in tinylcm_config and tinylcm_config["feature_transformation"].get("enabled", False):
            transform_config = tinylcm_config["feature_transformation"]
            
            if transform_config["type"] == "StandardScalerPCA":
                logger.info(f"Initializing StandardScalerPCA transformer with processor: {transform_config['model_path']}")
                feature_transformer = StandardScalerPCATransformer(
                    processor_path=transform_config["model_path"]
                )
                logger.info("StandardScalerPCA transformer initialized")
        else:
            logger.info("Feature transformation not enabled in config")
        
        # Initialize lightweight KNN
        knn_config = tinylcm_config["adaptive_classifier"]
        classifier = LightweightKNN(
            k=knn_config["k"],
            max_samples=knn_config["max_samples"],
            distance_metric=knn_config["distance_metric"],
            weight_by_distance=knn_config.get("weight_by_distance", False)
        )
        
        # Load initial KNN state
        initial_state_path_str = knn_config.get("initial_state_path")
        loaded_initial_state = False
        
        if initial_state_path_str:
            initial_state_path = Path(initial_state_path_str)
            if initial_state_path.exists():
                logger.info(f"Loading initial KNN state from: {initial_state_path}")
                try:
                    with open(initial_state_path, 'r') as f:
                        loaded_state_data = json.load(f)
                    
                    if "classifier" in loaded_state_data and isinstance(loaded_state_data["classifier"], dict):
                        classifier.set_state(loaded_state_data["classifier"])
                        logger.info(f"KNN initialized with {len(classifier.X_train)} samples")
                        loaded_initial_state = True
                    elif isinstance(loaded_state_data, dict) and "X_train" in loaded_state_data:
                        classifier.set_state(loaded_state_data)
                        logger.info(f"KNN initialized with {len(classifier.X_train)} samples")
                        loaded_initial_state = True
                    else:
                        logger.error("Invalid KNN state file format")
                except Exception as e:
                    logger.error(f"Error loading initial KNN state: {e}")
            else:
                logger.warning(f"Initial KNN state file not found: {initial_state_path}")
        
        # Fail fast if no initial state loaded
        if not loaded_initial_state:
            logger.error("No initial KNN state loaded. Cannot start without initial training data.")
            logger.error("Please provide initial_state_path in config or run create_initial_state.py first.")
            sys.exit(1)

        # Debug-Info zum Start: IO-Dtypes, Quantisierung, Feature-Dimensionen, Klassen
        try:
            mi = feature_extractor.get_model_info()
            in_dtype = None
            in_scale = None
            in_zero = None
            out_dtype = None
            feat_dim = None

            if mi.get("is_loaded") and mi.get("input_details"):
                idef = mi["input_details"][0]
                in_dtype = str(idef.get('dtype'))
                # quantization can be in 'quantization' or 'quantization_parameters'
                q = idef.get('quantization', None)
                if q and isinstance(q, (list, tuple)) and len(q) >= 2:
                    in_scale, in_zero = float(q[0] or 1.0), int(q[1] or 0)
                else:
                    qp = idef.get('quantization_parameters', {})
                    scales = qp.get('scales', []) if qp else []
                    zeros = qp.get('zero_points', []) if qp else []
                    if scales:
                        in_scale = float(scales[0])
                    if zeros:
                        in_zero = int(zeros[0])

            if mi.get("is_loaded") and mi.get("output_details"):
                out_list = mi["output_details"]
                oidx = feature_extractor.output_tensor_index
                if isinstance(oidx, int):
                    if oidx < 0:
                        oidx = len(out_list) + oidx
                else:
                    # Find by name if string
                    oidx = 0
                    for i, d in enumerate(out_list):
                        if d.get('name') == feature_extractor.output_tensor_index:
                            oidx = i
                            break
                odef = out_list[max(0, min(oidx, len(out_list)-1))]
                out_dtype = str(odef.get('dtype'))
                oshp = odef.get('shape')
                if oshp is not None and len(oshp) > 0:
                    feat_dim = int(oshp[-1])

            if feature_transformer is not None and getattr(feature_transformer, 'output_dim', None) is not None:
                feat_dim = int(feature_transformer.output_dim)

            class_count = len(classifier.get_classes())
            logger.info(
                f"Model IO: input_dtype={in_dtype}, input_scale/zero_point=({in_scale},{in_zero}), "
                f"output_dtype={out_dtype}, feature_dim={feat_dim}, classes={class_count}"
            )
        except Exception as e:
            logger.debug(f"Startup IO/feature logging failed: {e}")
        
        # Simple JSON logger for data (replacing DataLogger)
        data_logger_config = tinylcm_config["data_logger"]
        json_log_file = None
        if data_logger_config["enabled"]:
            log_dir = Path(data_logger_config["log_dir"])
            log_dir.mkdir(exist_ok=True)
            json_log_file = log_dir / f"data_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
            logger.info(f"Data logging to: {json_log_file}")
        frame_log_interval = int(data_logger_config.get("frame_log_interval", 10))
        
        rover.init(0) #Rover

        # Main inference loop
        logger.info("Starting inference loop")
        frame_count = 0
        last_inference_time = 0
        inference_interval = config["application"]["inference_interval_ms"] / 1000.0
        feature_dim_checked = False

        # Steering time
        last_steering_change_time = 0
        last_steering_action = None

        while running:
            try:
                # Check if it's time for next inference
                current_time = time.time()
                if current_time - last_inference_time < inference_interval:
                    time.sleep(0.01)
                    continue
                
                # Capture frame (wait briefly to avoid busy loop)
                frame = getattr(camera, 'wait_for_frame', camera.get_frame)(timeout=0.5) if hasattr(camera, 'wait_for_frame') else camera.get_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    # Throttle loop even on failure
                    last_inference_time = current_time
                    continue
                
                frame_count += 1
                last_inference_time = current_time
                                
                # Resize frame for inference
                inference_resolution = tuple(camera_config["inference_resolution"])
                resized_frames = crop_and_resize(frame, inference_resolution)
                
                # Process through pipeline
                start_time = time.time()

                # Multi-crop inference
                crop_predictions = []
                crop_confidences = []
                crop_drifts = []
                crop_drift_infos = []
                for idx, crop in enumerate(resized_frames):
                    # Preprocess and extract features for each crop
                    crop_features = feature_extractor.extract_features(crop)
                    if feature_transformer is not None:
                        crop_features = apply_feature_transformation(crop_features)

                    # Dimension check: runtime_dim == trained_dim (per Crop)
                    if not feature_dim_checked:
                        try:
                            curr_dim = int(crop_features.shape[-1]) if hasattr(crop_features, 'shape') else len(crop_features)
                            trained_dim = None
                            if getattr(classifier, 'X_train', None) is not None and len(classifier.X_train) > 0:
                                trained_dim = int(classifier.X_train.shape[1])
                            if trained_dim is not None and curr_dim != trained_dim:
                                logger.error(
                                    f"Feature dimension mismatch: runtime dim={curr_dim} vs trained dim={trained_dim}. "
                                    f"Ensure processor (.npz) and initial state use identical PCA/output dims. Exiting."
                                )
                                sys.exit(2)
                        except Exception as e:
                            logger.warning(f"Feature dimension sanity check failed: {e}")
                        finally:
                            feature_dim_checked = True

                    # Prediction for each crop
                    crop_prediction = "unknown"
                    crop_confidence = 0.0
                    try:
                        crop_probas = classifier.predict_proba(np.array([crop_features]))
                        if crop_probas is not None and len(crop_probas) > 0:
                            crop_confidence = np.max(crop_probas[0])
                            crop_classes = classifier.get_classes()
                            if len(crop_classes) > 0:
                                crop_max_idx = np.argmax(crop_probas[0])
                                crop_prediction = crop_classes[crop_max_idx]
                    except Exception as e:
                        logger.debug(f"Error getting crop prediction: {e}")
                        crop_prediction = "unknown"
                        crop_confidence = 0.0

                    crop_predictions.append(crop_prediction)
                    crop_confidences.append(crop_confidence)

                latency_ms = (time.time() - start_time) * 1000

                # Sample CPU/memory once per frame and reuse for all crop logs
                try:
                    cpu_now = psutil.cpu_percent(interval=None)
                except Exception:
                    cpu_now = None
                try:
                    mem_now = psutil.virtual_memory().percent
                except Exception:
                    mem_now = None

                # Rover steering logic
                left_obstacle = crop_predictions[0] == "obstacle"
                center_obstacle = crop_predictions[1] == "obstacle"
                right_obstacle = crop_predictions[2] == "obstacle"

                # Define steering actions
                steering_action = None
                if (left_obstacle and right_obstacle) or (center_obstacle and left_obstacle and right_obstacle):
                    steering_action = "stop"
                elif center_obstacle and left_obstacle:
                    steering_action = "spinRight"
                elif center_obstacle and right_obstacle:
                    steering_action = "spinLeft"
                elif center_obstacle:
                    steering_action = "spinLeft"
                elif left_obstacle:
                    steering_action = "spinRight"
                elif right_obstacle:
                    steering_action = "spinLeft"
                else:
                    steering_action = "forward"

                # Only change steering, if 2 seconds have passed and action changed
                if (steering_action != last_steering_action) or (current_time - last_steering_change_time >= 2.0):
                    try:
                        if steering_action == "stop":
                            logger.info("Rover: Obstacle left, right and center! Stopping..")
                            rover.stop()
                        elif steering_action == "spinRight":
                            logger.info("Rover: Obstacle left or center & left! Going right..")
                            rover.stop() #For experimentation to enable reproducability
                        elif steering_action == "spinLeft":
                            logger.info("Rover: Obstacle right or center & right! Going left..")
                            rover.stop() #For experimentation to enable reproducability
                        elif steering_action == "forward":
                            logger.info("Rover: No obstacle. Driving..")
                            rover.forward(speed)
                        last_steering_change_time = current_time
                        last_steering_action = steering_action
                    except Exception as e:
                        logger.error(f"Steering error: {e}", exc_info=True)


                # Per-frame logging of prediction, confidence, neighbor distances, and aggregated distance (jetzt für alle 3 crops)
                try:
                    if config.get("application", {}).get("log_each_prediction", True):
                        crop_logs = []
                        for i in range(3):
                            crop_pred = crop_predictions[i] if i < len(crop_predictions) else "unknown"
                            crop_conf = crop_confidences[i] if i < len(crop_confidences) else 0.0
                            crop_logs.append(
                                f"Crop {i}: Prediction={crop_pred}, Confidence={crop_conf:.2f}"
                            )
                        logger.info(
                            f"Frame {frame_count}: " + " | ".join(crop_logs) +
                            f", Latency={latency_ms:.1f}ms"
                        )
                except Exception:
                    pass

                if json_log_file and (frame_count % frame_log_interval == 0):
                    log_entry = {
                        "timestamp": time.time(),
                        "frame_count": frame_count,
                        "crop_predictions": crop_predictions,
                        "crop_confidences": [float(c) if c else 0.0 for c in crop_confidences],
                        "latency_ms": latency_ms
                    }
                    with open(json_log_file, 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')

                for i in range(len(crop_predictions)):
                    performance_logger.log_inference(
                        prediction=str(crop_predictions[i]),
                        confidence=float(crop_confidences[i]) if crop_confidences else 0.0,
                        latency_ms=latency_ms,
                        cpu_value=cpu_now,
                        memory_value=mem_now
                    )

                if frame_count % 100 == 0:
                    logger.info(
                        f"Frame {frame_count}: Crop predictions={crop_predictions}, "
                        f"Confidences={['{:.2f}'.format(c) for c in crop_confidences]}, "
                        f"Latency={latency_ms:.1f}ms"
                    )
                    performance_logger.save()
                    summary = performance_logger.get_summary()
                    logger.info(f"Performance summary: {summary}")
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                continue
        
        # Clean shutdown
        logger.info("Shutting down...")
        
        # Save final performance metrics
        if performance_logger:
            performance_logger.save()
            summary = performance_logger.get_summary()
            logger.info(f"Final performance summary: {summary}")
        
        # Cleanup
        camera.cleanup()
        rover.cleanup() #Rover
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
