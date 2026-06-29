"""
Prototype-based classifiers for embedded systems.

Both classifiers reduce the stored training set to a small number of
representative prototypes per class and reuse :class:`LightweightKNN`'s
distance and prediction machinery for inference. This keeps the on-device
runtime, the JSON state format, the KNNDistanceMonitor and the reference
statistics fully compatible with the existing k-NN path.

- ``NearestCentroidClassifier``: one prototype per class (the class mean).
  Prediction is the nearest class centroid; with inverse distance weighting
  the predicted probabilities form a smooth distribution over all centroids.
- ``KMeansPrototypeClassifier``: ``m`` prototypes per class (k-means cluster
  centers). Prediction is a (weighted) vote over the ``k`` nearest prototypes.

A k-means implementation in pure NumPy is included so the package keeps no
extra dependencies beyond NumPy (k-means only runs at training time on the
server; the device only loads the resulting prototypes via ``set_state``).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .knn import DistanceMetric, LightweightKNN

logger = logging.getLogger(__name__)


def kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    n_iter: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Compute k-means cluster centers using k-means++ init and Lloyd updates.

    Pure NumPy, no external dependencies. Returns at most ``n_clusters``
    centers (fewer if ``X`` has fewer rows). Designed for the small per-class
    sample counts seen in this project (N≈200, dim≈256).
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    n_clusters = int(min(n_clusters, n))
    if n_clusters <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float32)
    if n_clusters == 1:
        return X.mean(axis=0, keepdims=True).astype(np.float32)

    rng = np.random.RandomState(seed)

    # k-means++ initialization: spread initial centers by squared distance.
    centers = np.empty((n_clusters, X.shape[1]), dtype=np.float32)
    centers[0] = X[rng.randint(n)]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, n_clusters):
        total = float(closest_sq.sum())
        if total <= 1e-12:
            # All remaining points coincide with chosen centers; pad arbitrarily.
            centers[c] = X[rng.randint(n)]
        else:
            probs = closest_sq / total
            centers[c] = X[rng.choice(n, p=probs)]
        closest_sq = np.minimum(closest_sq, np.sum((X - centers[c]) ** 2, axis=1))

    # Lloyd iterations.
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(n_iter):
        # d[i, c] = squared distance of sample i to center c
        d = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            members = X[labels == c]
            if members.shape[0] > 0:
                centers[c] = members.mean(axis=0)
            # Empty clusters keep their previous center.

    return centers.astype(np.float32)


class NearestCentroidClassifier(LightweightKNN):
    """Nearest centroid classifier: one prototype (mean) per class.

    Stores a single centroid per class and predicts the class of the nearest
    centroid. ``k`` is set automatically to the number of centroids so that,
    with inverse distance weighting, the predicted probabilities span all
    classes while ``argmax`` still equals the nearest centroid.
    """

    def __init__(
        self,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        max_samples: int = 100,
        weight_by_distance: bool = True,
        idw_power: float = 1.0,
        idw_eps: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__(
            k=1,
            distance_metric=distance_metric,
            max_samples=max_samples,
            weight_by_distance=weight_by_distance,
            idw_power=idw_power,
            idw_eps=idw_eps,
        )

    def fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")
        y = np.asarray(labels)
        if y.shape[0] != X.shape[0]:
            raise ValueError("labels length must match features rows")

        classes = sorted(set(y.tolist()))
        protos = np.vstack([X[y == c].mean(axis=0) for c in classes]).astype(np.float32)
        # All centroids participate as neighbors so predict_proba covers all classes.
        self.k = len(classes)
        super().fit(protos, classes, timestamps=None)
        logger.info("NearestCentroid: built %d centroids from %d samples", len(classes), X.shape[0])

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["type"] = "NearestCentroidClassifier"
        return state


class KMeansPrototypeClassifier(LightweightKNN):
    """K-means prototype classifier: ``m`` prototypes per class.

    Runs k-means per class and stores the cluster centers as labeled
    prototypes. Prediction is a (distance-weighted) vote over the ``k``
    nearest prototypes.
    """

    def __init__(
        self,
        m_prototypes: int = 3,
        k: int = 1,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        max_samples: int = 100,
        weight_by_distance: bool = True,
        idw_power: float = 1.0,
        idw_eps: float = 1e-6,
        kmeans_iter: int = 50,
        kmeans_seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            k=k,
            distance_metric=distance_metric,
            max_samples=max_samples,
            weight_by_distance=weight_by_distance,
            idw_power=idw_power,
            idw_eps=idw_eps,
        )
        self.m_prototypes = int(m_prototypes)
        self.kmeans_iter = int(kmeans_iter)
        self.kmeans_seed = int(kmeans_seed)

    def fit(self, features: np.ndarray, labels: List[Any], timestamps: Optional[List[float]] = None) -> None:
        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")
        y = np.asarray(labels)
        if y.shape[0] != X.shape[0]:
            raise ValueError("labels length must match features rows")

        protos: List[np.ndarray] = []
        proto_labels: List[Any] = []
        for c in sorted(set(y.tolist())):
            Xc = X[y == c]
            centers = kmeans(Xc, self.m_prototypes, n_iter=self.kmeans_iter, seed=self.kmeans_seed)
            for center in centers:
                protos.append(center)
                proto_labels.append(c)
            logger.info("KMeans: class '%s' -> %d prototypes from %d samples", c, len(centers), Xc.shape[0])

        super().fit(np.vstack(protos).astype(np.float32), proto_labels, timestamps=None)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["type"] = "KMeansPrototypeClassifier"
        state["m_prototypes"] = self.m_prototypes
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state)
        self.m_prototypes = int(state.get("m_prototypes", self.m_prototypes))


# Map classifier "type" -> class for building from a config dict.
_CLASSIFIER_TYPES = {
    "LightweightKNN": LightweightKNN,
    "NearestCentroidClassifier": NearestCentroidClassifier,
    "KMeansPrototypeClassifier": KMeansPrototypeClassifier,
}


def build_classifier(config: Dict[str, Any]) -> LightweightKNN:
    """Build a classifier instance from an ``adaptive_classifier`` config dict.

    Selects the implementation via the ``type`` field (defaults to
    ``LightweightKNN`` for backward compatibility) and forwards the relevant
    hyperparameters. Unknown keys are ignored.
    """
    ctype = str(config.get("type", "LightweightKNN"))
    common = dict(
        distance_metric=config.get("distance_metric", "euclidean"),
        max_samples=config.get("max_samples", 200),
        weight_by_distance=config.get("weight_by_distance", False),
        idw_power=config.get("idw_power", 1.0),
    )

    if ctype == "NearestCentroidClassifier":
        return NearestCentroidClassifier(**common)
    if ctype == "KMeansPrototypeClassifier":
        return KMeansPrototypeClassifier(
            m_prototypes=config.get("m_prototypes", 3),
            k=config.get("k", 1),
            **common,
        )
    if ctype != "LightweightKNN":
        logger.warning("Unknown classifier type '%s', falling back to LightweightKNN", ctype)
    return LightweightKNN(k=config.get("k", 5), **common)
