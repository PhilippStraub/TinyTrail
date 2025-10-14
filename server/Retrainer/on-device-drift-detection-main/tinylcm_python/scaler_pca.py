"""
Feature transformation using StandardScaler and PCA.

Provides sklearn-free runtime transformation for embedded systems.
Loads pre-trained parameters from numpy arrays (.npz format) and
applies standardization followed by dimensionality reduction.
Maintains compatibility with legacy pickle models while avoiding
runtime dependencies on scikit-learn.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StandardScalerPCATransformer:
    def __init__(self, processor_path: str, *, strict: bool = True, allow_batch: bool = True, 
                 normalize_output: bool = False):
        self.processor_path = processor_path
        self.strict = strict
        self.allow_batch = allow_batch
        self.normalize_output = normalize_output

        self.mu: Optional[np.ndarray] = None
        self.inv_scale: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        self.explained_variance: Optional[float] = None
        self._load_arrays()

    def _load_arrays(self) -> None:
        if not os.path.exists(self.processor_path):
            raise FileNotFoundError(f"Processor artifact not found: {self.processor_path}")

        if self.processor_path.endswith('.npz'):
            with np.load(self.processor_path, allow_pickle=False) as data:
                self.mu = np.asarray(data['mu'], dtype=np.float32)
                self.inv_scale = np.asarray(data['inv_scale'], dtype=np.float32)
                self.W = np.asarray(data['W'], dtype=np.float32)
                self.input_dim = int(np.asarray(data['input_dim']).item())
                self.output_dim = int(np.asarray(data['output_dim']).item())
                if 'explained_variance' in data.files:
                    self.explained_variance = float(np.asarray(data['explained_variance']).item())
                else:
                    self.explained_variance = 0.0
        else:
            with open(self.processor_path, 'rb') as f:
                proc = pickle.load(f)
            scaler = proc.get('scaler')
            pca = proc.get('pca')
            if scaler is None:
                raise ValueError("Legacy processor missing 'scaler'")
            self.mu = np.asarray(scaler.mean_, dtype=np.float32)
            self.inv_scale = (1.0 / np.asarray(scaler.scale_, dtype=np.float32))
            if pca is not None:
                self.W = np.asarray(pca.components_, dtype=np.float32)
                self.output_dim = int(self.W.shape[0])
                self.explained_variance = float(np.sum(getattr(pca, 'explained_variance_ratio_', [0.0])) * 100.0)
            else:
                self.W = np.eye(len(self.mu), dtype=np.float32)
                self.output_dim = int(self.W.shape[0])
                self.explained_variance = 0.0
            self.input_dim = int(self.mu.shape[0])

        # Validation
        if self.mu is None or self.inv_scale is None or self.W is None:
            raise ValueError("Invalid processor artifact: missing arrays")
        if self.mu.shape != self.inv_scale.shape:
            raise ValueError("mu and inv_scale shape mismatch")
        if self.W.shape[1] != self.mu.shape[0]:
            raise ValueError("W second dimension must equal input_dim")

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mu is None or self.inv_scale is None or self.W is None:
            raise RuntimeError("Processor not loaded")

        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim == 2:
            if not self.allow_batch:
                raise ValueError("Batch input not allowed when allow_batch=False")
        else:
            raise ValueError(f"Expected 1D or 2D features, got shape {x.shape}")

        if self.strict and x.shape[1] != self.mu.shape[0]:
            raise ValueError(f"Dim mismatch: got {x.shape[1]}, expected {self.mu.shape[0]}")
        elif x.shape[1] != self.mu.shape[0]:
            D = self.mu.shape[0]
            if x.shape[1] > D:
                x = x[:, :D]
            else:
                pad = np.zeros((x.shape[0], D - x.shape[1]), dtype=x.dtype)
                x = np.hstack([x, pad])

        # Standardization: z = (x - μ) / σ
        z = (x - self.mu) * self.inv_scale
        # PCA projection: y = z @ W^T
        y = z @ self.W.T
        
        # Optional L2 normalization for cosine distance compatibility
        if self.normalize_output:
            # Handle both single sample and batch
            if y.ndim == 1:
                norm = np.linalg.norm(y)
                if norm > 1e-12:
                    y = y / norm
            else:
                norm = np.linalg.norm(y, axis=-1, keepdims=True)
                y = y / np.maximum(norm, 1e-12)

        return y[0] if features.ndim == 1 else y

    def get_state(self) -> Dict[str, Any]:
        return {
            "processor_path": self.processor_path,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "explained_variance": self.explained_variance,
            "strict": self.strict,
            "allow_batch": self.allow_batch,
            "normalize_output": self.normalize_output,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "processor_path" in state and state["processor_path"] != self.processor_path:
            self.processor_path = state["processor_path"]
            self._load_arrays()
        self.strict = state.get("strict", self.strict)
        self.allow_batch = state.get("allow_batch", self.allow_batch)
