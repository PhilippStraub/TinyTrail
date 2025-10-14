"""
TinyLCM Python - Lightweight Concept Monitoring for embedded systems.

Provides drift detection and adaptive learning capabilities optimized
for resource-constrained devices like Raspberry Pi Zero.
"""

from .drift_monitor import KNNDistanceMonitor
from .feature_extractor import TFLiteFeatureExtractor
from .knn import LightweightKNN, DistanceMetric
from .scaler_pca import StandardScalerPCATransformer
from .qparams import get_qparams

__version__ = "0.1.0"
__all__ = [
    "KNNDistanceMonitor",
    "TFLiteFeatureExtractor",
    "LightweightKNN",
    "DistanceMetric",
    "StandardScalerPCATransformer",
    "get_qparams",
]