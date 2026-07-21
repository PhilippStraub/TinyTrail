#!/usr/bin/env python3
"""
On-device image-folder inference and drift detection for Raspberry Pi Zero 2W.

This variant replaces the camera loop with a folder-backed image pipeline.
Images are read from a deployment-local input directory, processed one by one,
and passed through the same TFLite -> optional PCA -> LightweightKNN -> drift
monitor pipeline as before.

Expected input structure:
    ./input_images/<optional_class_name>/image.png
    ./input_images/image_001.jpg

If images are stored in per-class folders and
"ground_truth_from_parent_dir" is enabled in config, the parent folder name is
used as ground-truth label for basic accuracy logging.
"""

# --- BLAS/OpenMP thread pinning — MUST run before numpy/cv2/BLAS import ---
# On the Pi's ARM OpenBLAS an unpinned PCA projection matmul (1280 -> 608 on
# ImageNet/Imagenette) spins up ~3 native worker threads, roughly tripling
# process CPU and net power and corrupting the energy measurement (the
# "thread-count confound"). OpenBLAS reads these env vars at import time, so
# they MUST be set before numpy is first imported anywhere in the process.
import os
for _blas_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import argparse
import csv
import json
import logging
import os
import random as py_random
import psutil
import signal
import socket
import sys
import threading
import time
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.preprocessors import resize_image
from tinylcm_python.knn import LightweightKNN
from tinylcm_python.prototype_classifier import build_classifier
from tinylcm_python.drift_monitor import KNNDistanceMonitor
from tinylcm_python.feature_extractor import TFLiteFeatureExtractor
from tinylcm_python.scaler_pca import StandardScalerPCATransformer

logger = None
running = True
config = None
feature_transformer = None
process = psutil.Process(os.getpid())
performance_logger = None
resource_logger = None
IMG_SIZE = 224  # Default image size for feature extractor input (MobileNetV2)


class FolderImageSource:
    """Sequential image source backed by a folder on disk."""

    def __init__(
        self,
        input_dir: Path,
        recursive: bool = True,
        loop_forever: bool = False,
        supported_extensions: Optional[List[str]] = None,
        sort_mode: str = "name",
        ground_truth_from_parent_dir: bool = True,
        skip_non_image_files: bool = True,
        shuffle_files: bool = False,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.recursive = recursive
        self.loop_forever = loop_forever
        self.supported_extensions = {
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in (supported_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
        }
        self.sort_mode = sort_mode
        self.ground_truth_from_parent_dir = ground_truth_from_parent_dir
        self.skip_non_image_files = skip_non_image_files
        self.shuffle_files = shuffle_files
        self.shuffle_seed = shuffle_seed

        self.files = self._collect_files()
        self._shuffle_files_if_needed()

        self.index = 0

    def _collect_files(self) -> List[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")

        pattern_iter = self.input_dir.rglob('*') if self.recursive else self.input_dir.glob('*')
        files = []
        for path in pattern_iter:
            if not path.is_file():
                continue
            if path.suffix.lower() in self.supported_extensions:
                files.append(path)
            elif not self.skip_non_image_files:
                files.append(path)

        if self.sort_mode == 'mtime':
            files.sort(key=lambda p: p.stat().st_mtime)
        elif self.sort_mode == 'name':
            files.sort(key=lambda p: str(p).lower())
        else:
            files.sort(key=lambda p: str(p).lower())
        return files
    def _shuffle_files_if_needed(self) -> None:
        """
        Shuffles the collected image paths before inference.

        The original folder structure is not modified.
        If shuffle_seed is set, the shuffle order is reproducible.
        """
        if not self.shuffle_files:
            return

        rng = py_random.Random(self.shuffle_seed)
        rng.shuffle(self.files)

        if logger:
            if self.shuffle_seed is not None:
                logger.info(f"Shuffled {len(self.files)} input images with seed={self.shuffle_seed}")
            else:
                logger.info(f"Shuffled {len(self.files)} input images without fixed seed")

    def __len__(self) -> int:
        return len(self.files)

    def next_item(self) -> Optional[Dict[str, Any]]:
        if not self.files:
            return None
        if self.index >= len(self.files):
            if not self.loop_forever:
                return None
            self.index = 0

        path = self.files[self.index]
        self.index += 1

        image = cv2.imread(str(path))
        if image is None:
            logger.warning(f"Failed to read image: {path}")
            return {
                'path': path,
                'image': None,
                'ground_truth': self._infer_ground_truth(path),
            }

        return {
            'path': path,
            'image': image,
            'ground_truth': self._infer_ground_truth(path),
        }

    def _infer_ground_truth(self, path: Path) -> Optional[str]:
        if not self.ground_truth_from_parent_dir:
            return None
        try:
            rel = path.relative_to(self.input_dir)
        except ValueError:
            return None
        if len(rel.parts) <= 1:
            return None
        return rel.parts[-2]


class PerformanceLogger:
    """Performance logger for inference and drift metrics."""

    def __init__(
        self,
        log_dir: str = './logs',
        run_id: Optional[str] = None,
        run_start_time: Optional[float] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'performance_{self.run_id}.json'

        self.metrics: List[Dict[str, Any]] = []
        self.start_time = run_start_time or time.time()

        self.log_system_info()

    def _time_fields(self) -> Dict[str, Any]:
        now = time.time()
        return {
            'run_id': self.run_id,
            'timestamp_iso': datetime.now().astimezone().isoformat(timespec="milliseconds"),
            'timestamp_unix': round(now, 6),
            'elapsed_seconds': round(now - self.start_time, 3),
        }

    def log_system_info(self):
        import platform

        self.metrics.append({
            **self._time_fields(),
            'type': 'system_info',
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
        })

    def log_inference(
        self,
        image_path: str,
        prediction: str,
        confidence: float,
        latency_ms: float,
        drift_detected: bool = False,
        ground_truth: Optional[str] = None,
        correct: Optional[bool] = None,
        frame_count: Optional[int] = None,
        drift_info: Optional[Dict[str, Any]] = None,
    ):
        self.metrics.append({
            **self._time_fields(),
            'type': 'inference',
            'frame_count': frame_count,
            'image_path': image_path,
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'drift_detected': drift_detected,
            'drift_info': drift_info,
            'ground_truth': ground_truth,
            'correct': correct,

            # Snapshot directly after inference
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'process_rss_mb': process.memory_info().rss / (1024 * 1024),
        })

    def log_drift_event(self, detector_name: str, drift_type: str, details: Dict):
        self.metrics.append({
            **self._time_fields(),
            'type': 'drift_event',
            'detector': detector_name,
            'drift_type': drift_type,
            'details': details,
        })
        self.save()

    def save(self):
        def default_serializer(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, Path):
                return str(obj)
            return str(obj)

        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=default_serializer)

    def get_summary(self) -> Dict[str, Any]:
        inference_metrics = [m for m in self.metrics if m.get('type') == 'inference']
        drift_events = [m for m in self.metrics if m.get('type') == 'drift_event']

        if not inference_metrics:
            return {}

        latencies = [m['latency_ms'] for m in inference_metrics]
        confidences = [m['confidence'] for m in inference_metrics if m.get('confidence') is not None]
        labeled = [m for m in inference_metrics if m.get('correct') is not None]
        drifted_inferences = [m for m in inference_metrics if m.get('drift_detected')]

        summary = {
            'run_id': self.run_id,
            'total_inferences': len(inference_metrics),
            'total_drift_events': len(drift_events),
            'total_drifted_inferences': len(drifted_inferences),
            'avg_latency_ms': float(np.mean(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'runtime_seconds': time.time() - self.start_time,
        }

        if labeled:
            accuracy = sum(1 for item in labeled if item.get('correct')) / len(labeled)
            summary['labeled_samples'] = len(labeled)
            summary['accuracy'] = accuracy

        return summary


class ResourceCSVLogger:
    """
    Logs CPU and memory usage to CSV at a fixed frequency.
    Intended to run in the background during inference/training.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        frequency_hz: float = 5.0,
        run_id: Optional[str] = None,
        run_start_time: Optional[float] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"resource_usage_{self.run_id}.csv"

        self.frequency_hz = frequency_hz
        self.interval_seconds = 1.0 / frequency_hz

        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = run_start_time

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        if self.start_time is None:
            self.start_time = time.time()

        # psutil priming: first values are often meaningless without this
        psutil.cpu_percent(interval=None)
        process.cpu_percent(interval=None)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        if logger:
            logger.info(
                f"Resource CSV logging started: {self.csv_path} "
                f"at {self.frequency_hz} Hz"
            )

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if logger:
            logger.info("Resource CSV logging stopped")

    def _run(self):
        with open(self.csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                "run_id",
                "timestamp_iso",
                "timestamp_unix",
                "elapsed_seconds",
                "cpu_percent_total",
                "memory_percent_total",
                "memory_used_mb_total",
                "memory_available_mb_total",
                "process_cpu_percent",
                "process_rss_mb",
                "process_vms_mb",
            ])

            while not self.stop_event.is_set():
                now = time.time()
                elapsed = now - self.start_time if self.start_time else 0.0

                vm = psutil.virtual_memory()
                proc_mem = process.memory_info()

                writer.writerow([
                    self.run_id,
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    round(now, 6),
                    round(elapsed, 3),
                    psutil.cpu_percent(interval=None),
                    vm.percent,
                    round(vm.used / (1024 * 1024), 2),
                    round(vm.available / (1024 * 1024), 2),
                    process.cpu_percent(interval=None),
                    round(proc_mem.rss / (1024 * 1024), 2),
                    round(proc_mem.vms / (1024 * 1024), 2),
                ])

                csvfile.flush()
                time.sleep(self.interval_seconds)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_logging(
    log_level: str = 'INFO',
    log_dir: str = './logs',
    run_id: Optional[str] = None,
    console_log_level: str = 'ERROR',
) -> None:
    global logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    numeric_console_level = getattr(logging, console_log_level.upper(), logging.ERROR)

    logs_dir = Path(log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"drift_detection_{run_id}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_console_level)

    # force=True prevents duplicate handlers if the script is restarted in the same Python process.
    logging.basicConfig(
        level=min(numeric_level, numeric_console_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler],
        force=True,
    )

    logger = logging.getLogger('drift_detection')
    logger.info(f'Logging initialized at level {log_level}')
    logger.info(f'Log file: {log_file}')

    # Confound guard: confirm the BLAS/OpenMP thread pin (set in the module
    # header) actually took effect. If process_cpu_percent climbs well above
    # ~100% during PCA inference, OpenBLAS went multi-threaded and the energy
    # numbers are contaminated (the "thread-count confound").
    _openblas = os.environ.get('OPENBLAS_NUM_THREADS', 'unset')
    _omp = os.environ.get('OMP_NUM_THREADS', 'unset')
    logger.info(f'BLAS thread pin: OPENBLAS_NUM_THREADS={_openblas} OMP_NUM_THREADS={_omp} '
                f'(expect 1 — keeps PCA single-threaded for clean energy measurement)')
    if _openblas != '1' or _omp != '1':
        logger.warning('BLAS threads are NOT pinned to 1 — PCA net-energy per image will be '
                       'inflated by OpenBLAS multithreading (thread-count confound). '
                       'Do not trust the energy results from this run.')


def create_directory_structure(config: Dict) -> None:
    Path(config['tinylcm']['data_logger']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['image_input']['input_dir']).mkdir(parents=True, exist_ok=True)
    if config['application'].get('save_debug_frames', False):
        Path(config['application']['debug_output_dir']).mkdir(parents=True, exist_ok=True)


def on_drift_detected(drift_info: Dict[str, Any], record: Dict[str, Any]) -> None:
    detector_name = drift_info.get('detector_type', 'Unknown') if drift_info else 'Unknown'
    logger.debug(f'DRIFT DETECTED by {detector_name}: {drift_info}')
    log_drift_events = config['tinylcm']['data_logger'].get('log_drift_events', False)
    if performance_logger and drift_info is not None and log_drift_events:
        performance_logger.log_drift_event(detector_name, 'knn_distance', drift_info)


def handle_sigterm(signum, frame):
    global running
    if logger:
        logger.info(f'Received signal {signum}, shutting down gracefully...')
    running = False


def process_labels(labels_path: str) -> List[str]:
    with open(labels_path, 'r') as f:
        labels_raw = [line.strip() for line in f.readlines()]
    labels = []
    for label in labels_raw:
        parts = label.split(' ', 1)
        labels.append(parts[1] if len(parts) > 1 else label)
    return labels


def crop_top_960_if_needed(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == 1280 and img.shape[1] == 960:
        return img[:960, :]
    return img


def apply_feature_transformation(features: np.ndarray) -> np.ndarray:
    global feature_transformer
    if feature_transformer is None:
        return features
    try:
        return feature_transformer.transform(features)
    except Exception as e:
        logger.error(f'Error applying feature transformation: {e}')
        return features


def save_drift_image_if_needed(frame: np.ndarray, image_path: Path, frame_count: int) -> None:
    log_images = config['tinylcm']['data_logger'].get('log_images', False)
    if not log_images:
        return
    log_dir = Path(config['tinylcm']['data_logger']['log_dir'])
    image_save_path = log_dir / f"drift_frame_{frame_count:06d}_{image_path.stem}.jpg"
    try:
        cv2.imwrite(str(image_save_path), frame)
        logger.info(f'Drift frame saved: {image_save_path}')
    except Exception as e:
        logger.error(f'Failed to save drift image: {e}')


# def preprocess_bgr_for_model(image: np.ndarray) -> np.ndarray:
#     if image.ndim == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     elif image.ndim == 3 and image.shape[2] == 4:
#         image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
#     image = image.astype(np.float32)
#     image = (image / 127.5) - 1.0
#     return image


def preprocess_bgr_for_model(image: np.ndarray) -> np.ndarray:
    """
    Preprocessing for the pretrained MobileNetV2 feature extractor (Imagenette).

    Training used MobileNetV2 preprocess_input, i.e.:
        pixel values 0..255 -> -1.0..1.0  via (x / 127.5) - 1.0

    The client MUST match this exactly, otherwise the runtime features do not
    match the features the PCA / KNN state were built from.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0

    return image


def build_drift_detectors(tinylcm_config: Dict, classifier: LightweightKNN) -> List[KNNDistanceMonitor]:
    drift_detectors = []
    for detector_config in tinylcm_config.get('drift_detectors', []):
        if detector_config.get('type') != 'KNNDistanceMonitor':
            continue
        reference_stats_path = detector_config.get('reference_stats_path')
        reference_mean = None
        reference_std = None
        if reference_stats_path and Path(reference_stats_path).exists():
            with open(reference_stats_path, 'r') as f:
                ref_stats = json.load(f)
            reference_mean = ref_stats.get('reference_mean')
            reference_std = ref_stats.get('reference_std')
            logger.info(
                'Loaded reference stats: '
                f"mean={reference_mean}, std={reference_std}, "
                f"sample_count={ref_stats.get('sample_count')}"
            )
        drift_detector = KNNDistanceMonitor(
            delta=detector_config['delta'],
            lambda_threshold=detector_config['lambda_threshold'],
            exit_threshold_factor=detector_config.get('exit_threshold_factor', 0.7),
            high_confidence_threshold=detector_config.get('high_confidence_threshold', 0.9),
            stable_known_classes=detector_config.get('stable_known_classes', []),
            warm_up_samples=detector_config.get('warm_up_samples', 5),
            reference_update_interval=detector_config.get('reference_update_interval', 100),
            reference_update_factor=detector_config.get('reference_update_factor', 0.05),
            pause_reference_update_during_drift=detector_config.get('pause_reference_update_during_drift', True),
            drift_cooldown_period=detector_config.get('drift_cooldown_period', 30),
            initial_reference_mean=reference_mean,
            initial_reference_std=reference_std,
            reference_stats_path=reference_stats_path,
            use_adaptive_thresholds=detector_config.get('use_adaptive_thresholds', False),
            allow_runtime_threshold_adaptation=detector_config.get('allow_runtime_threshold_adaptation', False),
            adaptive_delta_std_multiplier=detector_config.get('adaptive_delta_std_multiplier', 1.5),
            adaptive_lambda_std_multiplier=detector_config.get('adaptive_lambda_std_multiplier', 5.0),
            aggregation_mode=detector_config.get('aggregation_mode', 'mean'),
            aggregation_q=detector_config.get('aggregation_q', 0.5),
            two_sided=detector_config.get('two_sided', True),
        )
        drift_detectors.append(drift_detector)
    return drift_detectors


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower == "true":
            return True
        if value_lower == "false":
            return False

    return None


def create_merged_analysis(
    performance_json_path: Path,
    resource_csv_path: Path,
    output_dir: str,
    run_id: str,
    tolerance_seconds: float = 0.5,
) -> Optional[Path]:
    """
    Creates a final merged CSV after inference.

    One row = one inference.
    Resource values are joined by nearest timestamp_unix.
    Accuracy is calculated if ground_truth/correct is available.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / f"merged_analysis_{run_id}.csv"

    if not performance_json_path.exists():
        if logger:
            logger.warning(f"Performance JSON not found: {performance_json_path}")
        return None

    if not resource_csv_path.exists():
        if logger:
            logger.warning(f"Resource CSV not found: {resource_csv_path}")
        return None

    with open(performance_json_path, "r") as f:
        performance_data = json.load(f)

    inference_rows = [
        item for item in performance_data
        if item.get("type") == "inference"
    ]

    if not inference_rows:
        if logger:
            logger.warning("No inference rows found. Merged analysis was not created.")
        return None

    resource_rows = []
    with open(resource_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            merge_time = _safe_float(row.get("timestamp_unix"))

            # Fallback for older files without timestamp_unix
            if merge_time is None:
                merge_time = _safe_float(row.get("elapsed_seconds"))

            if merge_time is None:
                continue

            row["_merge_time"] = merge_time
            resource_rows.append(row)

    resource_rows.sort(key=lambda r: r["_merge_time"])
    resource_times = [r["_merge_time"] for r in resource_rows]

    def nearest_resource_row(perf_row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        perf_time = _safe_float(perf_row.get("timestamp_unix"))

        # Fallback for older performance files
        if perf_time is None:
            perf_time = _safe_float(perf_row.get("timestamp"))

        if perf_time is None or not resource_rows:
            return None, None

        pos = bisect_left(resource_times, perf_time)

        candidates = []
        if pos < len(resource_rows):
            candidates.append(resource_rows[pos])
        if pos > 0:
            candidates.append(resource_rows[pos - 1])

        if not candidates:
            return None, None

        nearest = min(
            candidates,
            key=lambda r: abs(r["_merge_time"] - perf_time)
        )

        delta = abs(nearest["_merge_time"] - perf_time)

        if delta > tolerance_seconds:
            return None, delta

        return nearest, delta

    correct_values = [
        _safe_bool(row.get("correct"))
        for row in inference_rows
        if _safe_bool(row.get("correct")) is not None
    ]

    labeled_samples = len(correct_values)
    total_inferences = len(inference_rows)

    final_accuracy = None
    if labeled_samples > 0:
        final_accuracy = sum(1 for value in correct_values if value) / labeled_samples

    fieldnames = [
        "run_id",
        "frame_count",
        "timestamp_iso",
        "timestamp_unix",
        "elapsed_seconds",

        "image_path",
        "ground_truth",
        "prediction",
        "confidence",
        "correct",

        "cumulative_accuracy",
        "final_accuracy",
        "labeled_samples",
        "total_inferences",

        "latency_ms",
        "drift_detected",
        "drift_info_json",

        "resource_timestamp_iso",
        "resource_timestamp_unix",
        "resource_elapsed_seconds",
        "resource_time_delta_seconds",

        "cpu_percent_total",
        "memory_percent_total",
        "memory_used_mb_total",
        "memory_available_mb_total",
        "process_cpu_percent_resource",
        "process_rss_mb_resource",
        "process_vms_mb_resource",

        "cpu_percent_inference_snapshot",
        "memory_percent_inference_snapshot",
        "process_rss_mb_inference_snapshot",
    ]

    cumulative_labeled = 0
    cumulative_correct = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for perf_row in inference_rows:
            resource_row, resource_delta = nearest_resource_row(perf_row)

            correct_value = _safe_bool(perf_row.get("correct"))

            cumulative_accuracy = ""
            if correct_value is not None:
                cumulative_labeled += 1
                if correct_value:
                    cumulative_correct += 1
                cumulative_accuracy = round(cumulative_correct / cumulative_labeled, 4)

            writer.writerow({
                "run_id": perf_row.get("run_id", run_id),
                "frame_count": perf_row.get("frame_count", ""),
                "timestamp_iso": perf_row.get("timestamp_iso", ""),
                "timestamp_unix": perf_row.get("timestamp_unix", ""),
                "elapsed_seconds": perf_row.get("elapsed_seconds", ""),

                "image_path": perf_row.get("image_path", ""),
                "ground_truth": perf_row.get("ground_truth", ""),
                "prediction": perf_row.get("prediction", ""),
                "confidence": perf_row.get("confidence", ""),
                "correct": perf_row.get("correct", ""),

                "cumulative_accuracy": cumulative_accuracy,
                "final_accuracy": round(final_accuracy, 4) if final_accuracy is not None else "",
                "labeled_samples": labeled_samples,
                "total_inferences": total_inferences,

                "latency_ms": perf_row.get("latency_ms", ""),
                "drift_detected": perf_row.get("drift_detected", ""),
                "drift_info_json": json.dumps(
                    perf_row.get("drift_info"),
                    ensure_ascii=False,
                    default=str
                ) if perf_row.get("drift_info") is not None else "",

                "resource_timestamp_iso": resource_row.get("timestamp_iso", "") if resource_row else "",
                "resource_timestamp_unix": resource_row.get("timestamp_unix", "") if resource_row else "",
                "resource_elapsed_seconds": resource_row.get("elapsed_seconds", "") if resource_row else "",
                "resource_time_delta_seconds": round(resource_delta, 4) if resource_delta is not None else "",

                "cpu_percent_total": resource_row.get("cpu_percent_total", "") if resource_row else "",
                "memory_percent_total": resource_row.get("memory_percent_total", "") if resource_row else "",
                "memory_used_mb_total": resource_row.get("memory_used_mb_total", "") if resource_row else "",
                "memory_available_mb_total": resource_row.get("memory_available_mb_total", "") if resource_row else "",
                "process_cpu_percent_resource": resource_row.get("process_cpu_percent", "") if resource_row else "",
                "process_rss_mb_resource": resource_row.get("process_rss_mb", "") if resource_row else "",
                "process_vms_mb_resource": resource_row.get("process_vms_mb", "") if resource_row else "",

                "cpu_percent_inference_snapshot": perf_row.get("cpu_percent", ""),
                "memory_percent_inference_snapshot": perf_row.get("memory_percent", ""),
                "process_rss_mb_inference_snapshot": perf_row.get("process_rss_mb", ""),
            })

    if logger:
        logger.info(f"Merged analysis created: {output_path}")
        if final_accuracy is not None:
            logger.info(
                f"Final accuracy: {final_accuracy:.4f} "
                f"based on {labeled_samples}/{total_inferences} labeled samples"
            )
        else:
            logger.info("Final accuracy not available because no ground-truth labels were found.")

    return output_path


def main() -> None:
    global running, config, feature_transformer, performance_logger, resource_logger

    parser = argparse.ArgumentParser(description='Folder-based On-Device Drift Detection')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_start_time = time.time()
    log_dir = config['tinylcm']['data_logger']['log_dir']

    setup_logging(
        log_level=config['device']['log_level'],
        log_dir=log_dir,
        run_id=run_id,
        console_log_level=config['device'].get('console_log_level', 'ERROR'),
    )

    create_directory_structure(config)

    performance_logger = PerformanceLogger(
        log_dir=log_dir,
        run_id=run_id,
        run_start_time=run_start_time,
    )

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    logger.info('Starting folder-based On-Device Drift Detection')
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Device ID: {socket.gethostname() or 'drift-detector-001'}")

    image_cfg = config['image_input']
    image_source = FolderImageSource(
        input_dir=Path(image_cfg['input_dir']),
        recursive=image_cfg.get('recursive', True),
        loop_forever=image_cfg.get('loop_forever', False),
        supported_extensions=image_cfg.get('supported_extensions', ['.jpg', '.jpeg', '.png']),
        sort_mode=image_cfg.get('sort_mode', 'name'),
        ground_truth_from_parent_dir=image_cfg.get('ground_truth_from_parent_dir', True),
        skip_non_image_files=image_cfg.get('skip_non_image_files', True),
        shuffle_files=image_cfg.get('shuffle_files', False),
        shuffle_seed=image_cfg.get('shuffle_seed', None),
    )
    if len(image_source) == 0:
        logger.error(f"No input images found in {image_cfg['input_dir']}")
        sys.exit(1)
    logger.info(f"Discovered {len(image_source)} input images in {image_cfg['input_dir']}")

    labels = process_labels(config['model']['labels_path'])
    logger.info(f'Loaded {len(labels)} labels: {labels}')

    tinylcm_config = config['tinylcm']
    feature_extractor_config = tinylcm_config['feature_extractor']
    feature_extractor = TFLiteFeatureExtractor(
        model_path=feature_extractor_config['model_path'],
        feature_layer_index=feature_extractor_config['feature_layer_index'],
        normalize_features=feature_extractor_config.get('normalize_features', False),
        lazy_loading=feature_extractor_config.get('lazy_loading', False),
        preprocessors=[preprocess_bgr_for_model],
    )

    if tinylcm_config.get('feature_transformation', {}).get('enabled', False):
        transform_config = tinylcm_config['feature_transformation']
        if transform_config['type'] == 'StandardScalerPCA':
            feature_transformer = StandardScalerPCATransformer(processor_path=transform_config['model_path'])
            logger.info('StandardScalerPCA transformer initialized')

    knn_config = tinylcm_config['adaptive_classifier']
    # Wählt den Klassifikator anhand von adaptive_classifier.type
    # (LightweightKNN / NearestCentroidClassifier / KMeansPrototypeClassifier).
    classifier = build_classifier(knn_config)

    initial_state_path = Path(knn_config['initial_state_path'])
    if not initial_state_path.exists():
        logger.error(f'Initial KNN state file not found: {initial_state_path}')
        sys.exit(1)
    with open(initial_state_path, 'r') as f:
        loaded_state_data = json.load(f)
    if 'classifier' in loaded_state_data:
        classifier.set_state(loaded_state_data['classifier'])
    else:
        classifier.set_state(loaded_state_data)
    logger.info(f'KNN initialized with {len(classifier.X_train)} samples')

    drift_detectors = build_drift_detectors(tinylcm_config, classifier)
    logger.info(f'Initialized {len(drift_detectors)} drift detector(s)')

    inference_resolution = tuple(image_cfg.get('inference_resolution', [IMG_SIZE, IMG_SIZE]))
    per_image_delay_ms = int(image_cfg.get('delay_ms', config['application'].get('inference_interval_ms', 200)))
    frame_log_interval = tinylcm_config['data_logger'].get('frame_log_interval', 10)
    verbose_frame_logging = tinylcm_config['data_logger'].get('verbose_frame_logging', False)
    json_log_file = Path(log_dir) / f"stream_{run_id}.jsonl"
    write_jsonl = bool(tinylcm_config['data_logger'].get('enabled', True))
    baseline_mode = bool(config.get('application', {}).get('baseline_mode', False))
    if baseline_mode:
        logger.info(
            'BASELINE MODE: feature-extraction/classifier calls are skipped this run. '
            'Setup, delay_ms pacing, and resource/power logging are unchanged — only the '
            'compute call itself is omitted, to measure warm-idle power/CPU/RAM.'
        )

    frame_count = 0
    feature_dim_checked = False

    # Start resource logging only when the actual processing starts.
    resource_logger = ResourceCSVLogger(
        log_dir=log_dir,
        frequency_hz=5.0,
        run_id=run_id,
        run_start_time=run_start_time,
    )
    resource_logger.start()

    try:
        while running:
            item = image_source.next_item()
            if item is None:
                logger.info('No more input images. Stopping.')
                break

            image_path: Path = item['path']
            frame = item['image']
            ground_truth = item.get('ground_truth')

            if frame is None:
                continue

            frame_count += 1
            frame = crop_top_960_if_needed(frame)
            resized_frame = resize_image(frame, inference_resolution, False)
            inference_start_time = time.time()

            if baseline_mode:
                features = None
                prediction = 'baseline'
                confidence = 0.0
            else:
                features = feature_extractor.extract_features(resized_frame)
                if feature_transformer is not None:
                    features = apply_feature_transformation(features)

                if not feature_dim_checked:
                    curr_dim = int(features.shape[-1]) if hasattr(features, 'shape') else len(features)
                    trained_dim = (
                        int(classifier.X_train.shape[1])
                        if getattr(classifier, 'X_train', None) is not None and len(classifier.X_train) > 0
                        else None
                    )
                    if trained_dim is not None and curr_dim != trained_dim:
                        logger.error(
                            f'Feature dimension mismatch: runtime dim={curr_dim} vs trained dim={trained_dim}. '
                            'Ensure processor (.npz) and initial state use identical PCA/output dims.'
                        )
                        sys.exit(2)
                    feature_dim_checked = True

                prediction = 'unknown'
                confidence = 0.0
                try:
                    probas = classifier.predict_proba(np.array([features]))
                    if probas is not None and len(probas) > 0:
                        confidence = float(np.max(probas[0]))
                        classes = classifier.get_classes()
                        if len(classes) > 0:
                            prediction = classes[int(np.argmax(probas[0]))]
                except Exception as e:
                    logger.debug(f'Error getting prediction: {e}')

            drift_detected = False
            drift_info = None
            drift_distances = None
            detected_record = None

            if drift_detectors:
                ref_labels = set(drift_detectors[0].stable_known_classes or [])
                if ref_labels and classifier.X_train is not None:
                    d_all = classifier._compute_distances(features)
                    mask = np.isin(classifier.y_train, list(ref_labels))
                    d_sel = d_all[mask]
                    if d_sel.shape[0] > 0:
                        kk = min(classifier.k, d_sel.shape[0])
                        idx = np.argpartition(d_sel, kk - 1)[:kk]
                        if str(classifier.distance_metric).lower().endswith('euclidean'):
                            dk = np.sqrt(np.maximum(d_sel[idx], 0.0))
                        else:
                            dk = d_sel[idx]
                        drift_distances = dk.tolist()

            for drift_detector in drift_detectors:
                record = {
                    'features': features,
                    'prediction': prediction,
                    'confidence': confidence,
                    'classifier': classifier,
                    '_knn_distances': drift_distances if drift_distances is not None else getattr(classifier, '_last_distances', None),
                    'image_path': str(image_path),
                    'ground_truth': ground_truth,
                }
                detected, info = drift_detector.update(record)
                if detected:
                    drift_detected = True
                    drift_info = info
                    detected_record = record
                    break

            latency_ms = (time.time() - inference_start_time) * 1000.0
            correct = (prediction == ground_truth) if ground_truth is not None else None

            if verbose_frame_logging:
                neigh = getattr(classifier, '_last_distances', []) or []
                dists_fmt = ', '.join(
                    f'{float(d):.2f}' for d in (neigh[: classifier.k] if hasattr(classifier, 'k') else neigh)
                )
                dists_str = f'[{dists_fmt}]' if dists_fmt else '[]'
                agg_dist = None
                ref_mean = None
                if drift_detectors:
                    dd = drift_detectors[0]
                    hist = getattr(dd, 'distance_history', None)
                    if hist and len(hist) > 0:
                        agg_dist = float(list(hist)[-1])
                    ref_mean = getattr(dd, 'reference_mean', None)

                logger.info(
                    f"Image {frame_count}/{len(image_source)}: path={image_path.name}, gt={ground_truth}, "
                    f"pred={prediction}, conf={confidence:.2f}, correct={correct}, kNN_dists={dists_str}, "
                    f"AggDist={(agg_dist if agg_dist is not None else float('nan')):.2f}, "
                    f"RefMean={(ref_mean if ref_mean is not None else float('nan')):.2f}, "
                    f"Latency={latency_ms:.1f}ms, Drift={drift_detected}"
                )

            if drift_detected:
                if detected_record is not None:
                    on_drift_detected(drift_info, detected_record)
                save_drift_image_if_needed(frame, image_path, frame_count)

            if write_jsonl and (frame_count % frame_log_interval == 0 or drift_detected):
                now_log = time.time()
                with open(json_log_file, 'a') as f:
                    f.write(json.dumps({
                        'run_id': run_id,
                        'timestamp_iso': datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        'timestamp_unix': round(now_log, 6),
                        'elapsed_seconds': round(now_log - run_start_time, 3),
                        'frame_count': frame_count,
                        'image_path': str(image_path),
                        'ground_truth': ground_truth,
                        'prediction': prediction,
                        'confidence': confidence,
                        'latency_ms': latency_ms,
                        'drift_detected': drift_detected,
                        'drift_info': drift_info,
                        'correct': correct,
                    }, default=str) + '\n')

            performance_logger.log_inference(
                image_path=str(image_path),
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                drift_detected=drift_detected,
                ground_truth=ground_truth,
                correct=correct,
                frame_count=frame_count,
                drift_info=drift_info,
            )

            if frame_count % 100 == 0 or frame_count == len(image_source):
                performance_logger.save()
                logger.debug(f'Performance summary: {performance_logger.get_summary()}')

            if per_image_delay_ms > 0:
                time.sleep(per_image_delay_ms / 1000.0)

    finally:
        if resource_logger:
            resource_logger.stop()

        if performance_logger:
            performance_logger.save()

        merged_path = None
        if performance_logger and resource_logger:
            merged_path = create_merged_analysis(
                performance_json_path=performance_logger.log_file,
                resource_csv_path=resource_logger.csv_path,
                output_dir=log_dir,
                run_id=run_id,
                tolerance_seconds=0.5,
            )

        summary = performance_logger.get_summary() if performance_logger else {}

        logger.info(f'Final performance summary: {summary}')

        if merged_path:
            logger.info(f'Final merged analysis file: {merged_path}')

        logger.info('Shutdown complete')


if __name__ == '__main__':
    main()
