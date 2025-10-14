"""
Lightweight k-nearest neighbors classifier for embedded systems.

Implements brute-force KNN with inverse distance weighting and
temporal tie-breaking. Optimized for small datasets (N≈200) on
resource-constrained devices like Raspberry Pi Zero.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class LightweightKNN:
    """Lightweight k-nearest neighbors classifier optimized for embedded systems.
    
    Implements efficient KNN classification with optional inverse distance weighting
    and temporal tie-breaking. Designed for real-time drift detection on resource-
    constrained devices like Raspberry Pi Zero.
    
    Attributes:
        k: Number of nearest neighbors to consider
        distance_metric: Distance metric ('euclidean' or 'cosine')
        max_samples: Maximum training samples to retain (FIFO eviction)
        weight_by_distance: Use inverse distance weighting for predictions
        tie_break_by_time: Prefer newer samples when distances are equal
    """

    def __init__(
        self,
        k: int = 5,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        max_samples: int = 100,
        weight_by_distance: bool = False,
        tie_break_by_time: bool = True,
        idw_power: float = 1.0,
        idw_eps: float = 1e-6,
    ):
        self.k = int(k)
        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric(distance_metric.lower())
        self.distance_metric = distance_metric
        self.max_samples = int(max_samples)
        self.weight_by_distance = bool(weight_by_distance)
        self.tie_break_by_time = bool(tie_break_by_time)
        self.idw_power = float(idw_power)
        self.idw_eps = float(idw_eps)

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        self._classes: List[Any] = []
        self._class_to_idx: Dict[Any, int] = {}  # Cache for O(1) class lookups

        self._last_distances: List[float] = []
        self._last_neighbor_labels: List[Any] = []

        self._total_prediction_time = 0.0
        self._total_predictions = 0

    def fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")
        if len(labels) != X.shape[0]:
            raise ValueError("labels length must match features rows")
        y = np.asarray(labels)
        t = np.asarray(timestamps if timestamps is not None else [time.time()] * X.shape[0], dtype=np.float64)

        if X.shape[0] > self.max_samples:
            idx = np.argsort(t)[-self.max_samples:]
            X, y, t = X[idx], y[idx], t[idx]

        self.X_train = np.ascontiguousarray(X, dtype=np.float32)
        self.y_train = y
        self.timestamps = t
        self._classes = sorted(list(set(y.tolist())))
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

    def _append_sample(self, x: np.ndarray, y: Any, ts: float) -> None:
        if self.X_train is None:
            self.X_train = x[None, :].astype(np.float32)
            self.y_train = np.array([y], dtype=object)
            self.timestamps = np.array([ts], dtype=np.float64)
        else:
            self.X_train = np.vstack([self.X_train, x[None, :].astype(np.float32)])
            self.y_train = np.append(self.y_train, y)
            self.timestamps = np.append(self.timestamps, ts)

        if self.X_train.shape[0] > self.max_samples:
            oldest = np.argmin(self.timestamps)
            self.X_train = np.delete(self.X_train, oldest, axis=0)
            self.y_train = np.delete(self.y_train, oldest, axis=0)
            self.timestamps = np.delete(self.timestamps, oldest, axis=0)

    def add_sample(self, feature: np.ndarray, label: Any, timestamp: Optional[float] = None) -> None:
        x = np.asarray(feature, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("feature must be 1D")
        ts = float(timestamp if timestamp is not None else time.time())
        self._append_sample(x, label, ts)
        self._classes = sorted(list(set(self.y_train.tolist()))) if self.y_train is not None else []
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

    def incremental_fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")
        if len(y) != X.shape[0]:
            raise ValueError("labels length must match features rows")
        if timestamps is None:
            t = [time.time()] * X.shape[0]
        else:
            if len(timestamps) != X.shape[0]:
                raise ValueError("timestamps length must match features rows")
            t = timestamps
        for i in range(X.shape[0]):
            self._append_sample(X[i], y[i], float(t[i]))
        self._classes = sorted(list(set(self.y_train.tolist()))) if self.y_train is not None else []
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

    def _compute_distances(self, q: np.ndarray) -> np.ndarray:
        X = self.X_train
        if X is None or X.shape[0] == 0:
            return np.array([], dtype=np.float32)
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            diff = X - q[None, :]
            # L2²: d²[i] = Σⱼ(X[i,j] - q[j])²
            d2 = np.einsum('ij,ij->i', diff, diff)
            return d2
        elif self.distance_metric == DistanceMetric.COSINE:
            qn = np.linalg.norm(q) + 1e-12
            Xn = np.linalg.norm(X, axis=1) + 1e-12
            # Cosine: sim = (X·q) / (||X|| * ||q||)
            sim = (X @ q) / (Xn * qn)
            sim = np.clip(sim, -1.0, 1.0)
            return 1.0 - sim
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _find_neighbors(self, q: np.ndarray) -> List[Tuple[int, float]]:
        if self.X_train is None or self.X_train.shape[0] == 0:
            return []
        distances = self._compute_distances(q)
        k = min(self.k, distances.shape[0])
        
        if k < distances.shape[0]:
            idx = np.argpartition(distances, k)[:k]
            if self.tie_break_by_time and self.timestamps is not None:
                # Lexsort: secondary sort by -timestamp
                sorted_order = np.lexsort((-self.timestamps[idx], distances[idx]))
                idx = idx[sorted_order]
            else:
                sorted_order = np.argsort(distances[idx])
                idx = idx[sorted_order]
        else:
            if self.tie_break_by_time and self.timestamps is not None:
                idx = np.lexsort((-self.timestamps, distances))
            else:
                idx = np.argsort(distances)
            idx = idx[:k]
        
        dsel = distances[idx]
        self._last_distances = np.sqrt(dsel).tolist() if self.distance_metric == DistanceMetric.EUCLIDEAN else dsel.tolist()
        self._last_neighbor_labels = self.y_train[idx].tolist()
        return list(zip(idx.tolist(), dsel.tolist()))

    def predict(self, features: np.ndarray) -> List[Any]:
        start = time.time()
        if self.X_train is None or self.X_train.shape[0] == 0:
            return ["unknown"] * (1 if features.ndim == 1 else features.shape[0])
        Xq = np.asarray(features, dtype=np.float32)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        preds: List[Any] = []
        for i in range(Xq.shape[0]):
            neigh = self._find_neighbors(Xq[i])
            if self.weight_by_distance:
                weights: Dict[Any, float] = {}
                for idx, d in neigh:
                    # IDW: w = 1 / (ε + d^p)
                    w = 1.0 / (self.idw_eps + (np.sqrt(d) ** self.idw_power))
                    c = self.y_train[idx]
                    weights[c] = weights.get(c, 0.0) + w
                pred = max(weights.items(), key=lambda kv: kv[1])[0]
            else:
                labels = [self.y_train[idx] for idx, _ in neigh]
                values, counts = np.unique(labels, return_counts=True)
                pred = values[np.argmax(counts)]
            preds.append(pred)
        self._total_prediction_time += time.time() - start
        self._total_predictions += Xq.shape[0]
        return preds

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.X_train.shape[0] == 0:
            n = 1 if features.ndim == 1 else features.shape[0]
            # Return uniform distribution if classes are known, else single column
            if self._classes:
                c = len(self._classes)
                return np.full((n, c), 1.0/c, dtype=np.float32)
            else:
                return np.ones((n, 1), dtype=np.float32)
        Xq = np.asarray(features, dtype=np.float32)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        classes = np.array(self._classes, dtype=object)
        C = len(classes)
        probas = np.zeros((Xq.shape[0], C), dtype=np.float32)
        for i in range(Xq.shape[0]):
            neigh = self._find_neighbors(Xq[i])
            if self.weight_by_distance:
                weights = np.zeros(C, dtype=np.float32)
                for idx, d in neigh:
                    w = 1.0 / (self.idw_eps + (np.sqrt(d) ** self.idw_power))
                    c = self.y_train[idx]
                    ci = self._class_to_idx[c]  # Map class to index using cached dictionary
                    weights[ci] += w
                s = weights.sum()
                probas[i] = weights / s if s > 0 else np.full(C, 1.0 / C, dtype=np.float32)
            else:
                counts = np.zeros(C, dtype=np.float32)
                for idx, _ in neigh:
                    c = self.y_train[idx]
                    ci = self._class_to_idx[c]  # Map class to index using cached dictionary
                    counts[ci] += 1.0
                probas[i] = counts / counts.sum()
        return probas

    def get_performance_metrics(self) -> Dict[str, float]:
        avg_time = 0.0 if self._total_predictions == 0 else self._total_prediction_time / self._total_predictions
        return {
            "total_prediction_time": self._total_prediction_time,
            "total_predictions": self._total_predictions,
            "avg_prediction_time": avg_time,
            "sample_count": 0 if self.X_train is None else int(self.X_train.shape[0]),
            "class_count": len(self._classes),
        }

    def get_state(self) -> Dict[str, Any]:
        X = None if self.X_train is None else self.X_train.tolist()
        y = None if self.y_train is None else self.y_train.tolist()
        t = None if self.timestamps is None else self.timestamps.tolist()
        return {
            "type": "LightweightKNN",
            "k": self.k,
            "distance_metric": self.distance_metric.value,
            "max_samples": self.max_samples,
            "weight_by_distance": self.weight_by_distance,
            "tie_break_by_time": self.tie_break_by_time,
            "idw_power": self.idw_power,
            "idw_eps": self.idw_eps,
            "X_train": X,
            "y_train": y,
            "timestamps": t,
            "classes": list(self._classes),
            "metrics": {
                "total_prediction_time": self._total_prediction_time,
                "total_predictions": self._total_predictions,
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.k = state.get("k", self.k)
        metric = state.get("distance_metric", self.distance_metric)
        if isinstance(metric, str):
            self.distance_metric = DistanceMetric(metric.lower())
        else:
            self.distance_metric = metric
        self.max_samples = state.get("max_samples", self.max_samples)
        self.weight_by_distance = state.get("weight_by_distance", self.weight_by_distance)
        self.tie_break_by_time = state.get("tie_break_by_time", self.tie_break_by_time)
        self.idw_power = state.get("idw_power", self.idw_power)
        self.idw_eps = state.get("idw_eps", self.idw_eps)

        X = state.get("X_train")
        y = state.get("y_train")
        t = state.get("timestamps")
        self.X_train = None if X is None else np.ascontiguousarray(np.array(X, dtype=np.float32))
        self.y_train = None if y is None else np.array(y, dtype=object)
        self.timestamps = None if t is None else np.array(t, dtype=np.float64)
        # Restore classes; if missing or empty, derive from y_train
        classes_from_state = state.get("classes", [])
        if classes_from_state:
            self._classes = list(classes_from_state)
        elif self.y_train is not None and len(self.y_train) > 0:
            self._classes = sorted(list(set(self.y_train.tolist())))
        else:
            self._classes = []
        # Rebuild class->index mapping after loading state
        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}
        metrics = state.get("metrics", {})
        self._total_prediction_time = metrics.get("total_prediction_time", 0.0)
        self._total_predictions = metrics.get("total_predictions", 0)

    def get_samples(self) -> Tuple[List[Any], List[Any], List[float]]:
        X = [] if self.X_train is None else self.X_train.tolist()
        y = [] if self.y_train is None else self.y_train.tolist()
        t = [] if self.timestamps is None else self.timestamps.tolist()
        return X, y, t

    def get_classes(self) -> List[Any]:
        return list(self._classes)

    def get_last_neighbor_labels(self) -> List[Any]:
        return self._last_neighbor_labels

    def clear(self) -> None:
        self.X_train = None
        self.y_train = None
        self.timestamps = None
        self._classes = []
        logger.debug("Cleared all training samples from classifier")
