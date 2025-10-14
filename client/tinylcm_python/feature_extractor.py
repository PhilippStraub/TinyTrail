"""
TensorFlow Lite feature extraction for on-device inference.

Extracts intermediate layer features from quantized TFLite models,
handling INT8 quantization/dequantization transparently. Optimized
for Raspberry Pi Zero 2W with minimal memory footprint.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import os
import time
import threading
import logging

from tflite_runtime.interpreter import Interpreter
from .qparams import get_qparams

logger = logging.getLogger(__name__)


class TFLiteFeatureExtractor:
    """TensorFlow Lite model wrapper for feature extraction.
    
    Extracts intermediate layer outputs from quantized TFLite models,
    typically from the global average pooling layer before classification.
    Optimized for embedded devices with support for lazy loading and
    thread-safe inference.
    
    Designed for real-time feature extraction on Raspberry Pi Zero 2W
    with minimal memory footprint and CPU usage.
    """
    
    def __init__(
        self,
        model_path: str,
        feature_layer_index: int = -1,
        input_tensor_index: int = 0,
        output_tensor_index: Optional[int] = None,
        preprocessors: Optional[List[Callable]] = None,
        normalize_features: bool = False,
        lazy_loading: bool = True,
        input_shape: Optional[List[int]] = None,
        input_dtype: str = 'float32',
        thread_safe: bool = True
    ):
        """Initialize the TFLite feature extractor.
        
        Args:
            model_path: Path to the TFLite model file
            feature_layer_index: Index of the layer to extract features from
                               (default: -1, the last layer/output)
            input_tensor_index: Index of the input tensor (default: 0)
            output_tensor_index: Index of the output tensor (default: determined by feature_layer_index)
            preprocessors: List of preprocessing functions to apply to the input
            normalize_features: Whether to normalize feature vectors to unit length
            lazy_loading: Whether to load the model only when needed (saves memory)
            input_shape: Expected input shape (used for validation), e.g. [1, 224, 224, 3]
            input_dtype: Expected input data type ('float32', 'uint8', etc.)
            thread_safe: Whether to use thread locking for multi-threaded safety
        """
        self.model_path = model_path
        self.feature_layer_index = feature_layer_index
        self.input_tensor_index = input_tensor_index
        self.output_tensor_index = output_tensor_index
        self.preprocessors = preprocessors or []
        self.normalize_features = normalize_features
        self.lazy_loading = lazy_loading
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.thread_safe = thread_safe
        
        # Model loading state
        self._interpreter: Optional[Interpreter] = None
        self._input_details = None
        self._output_details = None
        self._lock = threading.RLock() if thread_safe else None
        self._is_loading = False
        self._load_error = None

        # Inference statistics
        self._total_inference_time = 0
        self._inference_count = 0
        
        logger.debug(
            f"Initialized TFLiteFeatureExtractor with model={model_path}, "
            f"feature_layer_index={feature_layer_index}, lazy_loading={lazy_loading}"
        )
        
        # Load model immediately if lazy loading disabled
        if not lazy_loading:
            self._load_interpreter()

    def _load_interpreter(self) -> None:
        """Load the TFLite interpreter.
        
        This method loads the TensorFlow Lite model and allocates tensors.
        If thread_safe is True, this method is protected by a lock.
        """
        if self._interpreter is not None:
            return
            
        if self.thread_safe:
            if self._is_loading:
                while self._is_loading and self._load_error is None:
                    time.sleep(0.01)
                    
                if self._load_error is not None:
                    raise self._load_error
                    
                return
                
            with self._lock:
                if self._interpreter is not None:
                    return  # Loaded by another thread
                self._is_loading = True
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"TFLite model file not found: {self.model_path}")
                
            self._interpreter = Interpreter(model_path=self.model_path)
            
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            
            # Set up for feature extraction - here's the key part for using output as features
            if self.output_tensor_index is None:
                # Default to using the specified feature layer index
                self.output_tensor_index = self.feature_layer_index
            
            # Check tensor indices
            if isinstance(self.output_tensor_index, int):
                if abs(self.output_tensor_index) >= len(self._output_details):
                    logger.warning(
                        f"Feature layer index {self.output_tensor_index} is out of range. "
                        f"Model has {len(self._output_details)} output tensors. "
                        f"Using the last output tensor."
                    )
                    self.output_tensor_index = -1
            
            # Validate input shape if provided
            if self.input_shape is not None:
                actual_shape = self._input_details[self.input_tensor_index]['shape']
                # Check that shapes are compatible (allowing for dynamic dimensions)
                if len(actual_shape) != len(self.input_shape):
                    logger.warning(
                        f"Input shape mismatch: expected {self.input_shape}, "
                        f"but model expects {actual_shape}"
                    )
                else:
                    for i, (expected, actual) in enumerate(zip(self.input_shape, actual_shape)):
                        if expected is not None and actual != -1 and expected != actual:
                            logger.warning(
                                f"Input shape mismatch at dimension {i}: "
                                f"expected {expected}, but model expects {actual}"
                            )
            
            logger.debug(f"TFLite interpreter loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {str(e)}")
            self._load_error = e
            raise
        finally:
            if self.thread_safe:
                self._is_loading = False

    def _ensure_interpreter_loaded(self) -> None:
        """Ensure the TFLite interpreter is loaded.
        
        This method checks if the interpreter is loaded and loads it if needed.
        """
        if self._interpreter is None:
            self._load_interpreter()

    def extract_features(self, input_data: Any) -> np.ndarray:
        """Extract features from the input data using the TFLite model.
        
        IMPORTANT: This is a synchronous, blocking operation. The execution time
        depends on the size of the input data, the complexity of the model, and
        the hardware capabilities of the device. For resource-constrained devices
        like the Raspberry Pi Zero, this operation may take significant time.
        
        If low-latency responses are required in your application, consider:
        1. Running this method in a separate thread
        2. Using a simpler model or pre-extracted features
        3. Batch processing inputs when possible
        
        Args:
            input_data: Input data in the format expected by the model
            
        Returns:
            Feature vector extracted from the specified layer
        """
        start_time = time.time()
        
        # Thread safety for inference
        if self.thread_safe:
            with self._lock:
                result = self._extract_features_internal(input_data)
        else:
            result = self._extract_features_internal(input_data)
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self._total_inference_time += inference_time
        self._inference_count += 1
        
        return result

    def _extract_features_internal(self, input_data: Any) -> np.ndarray:
        """Internal method to extract features.
        
        This method is called by extract_features and handles the actual feature extraction.
        """
        self._ensure_interpreter_loaded()
        
        processed_data = input_data
        for preprocessor in self.preprocessors:
            processed_data = preprocessor(processed_data)
        
        input_details = self._input_details[self.input_tensor_index]
        
        if not isinstance(processed_data, np.ndarray):
            processed_data = np.array(processed_data)

        expected_shape = input_details['shape']
        if processed_data.ndim == len(expected_shape) - 1:
            processed_data = np.expand_dims(processed_data, axis=0)

        in_dtype = input_details['dtype']
        scale, zero_point = get_qparams(input_details)

        if in_dtype in (np.int8, np.uint8):
            real = processed_data.astype(np.float32)
            # TFLite quantization: q = round(x/scale + zero_point)
            q = np.round(real / scale + zero_point)
            if in_dtype == np.int8:
                q = np.clip(q, -128, 127).astype(np.int8)
            else:
                q = np.clip(q, 0, 255).astype(np.uint8)
            processed = q
        else:
            processed = processed_data.astype(in_dtype)

        # Final shape check
        if tuple(processed.shape) != tuple(expected_shape):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Reshaping input from {processed.shape} to {expected_shape}")
            try:
                processed = processed.reshape(expected_shape)
            except Exception:
                raise ValueError(f"Input shape mismatch: {processed.shape} vs expected {expected_shape}")

        self._interpreter.set_tensor(input_details['index'], processed)
        self._interpreter.invoke()
        
        if isinstance(self.output_tensor_index, int):
            if self.output_tensor_index < 0:
                tensor_index = len(self._output_details) + self.output_tensor_index
            else:
                tensor_index = self.output_tensor_index
                
            output_details = self._output_details[tensor_index]
            feature_vector = self._interpreter.get_tensor(output_details['index'])
        else:
            output_details = None
            for details in self._output_details:
                if details['name'] == self.output_tensor_index:
                    output_details = details
                    feature_vector = self._interpreter.get_tensor(details['index'])
                    break
            else:
                raise ValueError(f"Tensor '{self.output_tensor_index}' not found in model outputs")
        
        out_details = output_details
        out_dtype = out_details.get('dtype', feature_vector.dtype)
        if out_dtype in (np.int8, np.uint8):
            o_scale, o_zero = get_qparams(out_details)
            feature_vector = (feature_vector.astype(np.float32) - o_zero) * o_scale

        if feature_vector.ndim > 1 and feature_vector.shape[0] == 1:
            feature_vector = feature_vector[0]
        
        if self.normalize_features:
            # L2 normalization: x_norm = x / ||x||₂
            norm = np.linalg.norm(feature_vector, ord=2, axis=-1, keepdims=True)
            norm = np.maximum(norm, 1e-12)
            feature_vector = feature_vector / norm
        
        return feature_vector

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the feature extractor.
        
        Returns:
            Dict containing the serializable state
        """
        return {
            "type": "TFLiteFeatureExtractor",
            "model_path": self.model_path,
            "feature_layer_index": self.feature_layer_index,
            "input_tensor_index": self.input_tensor_index,
            "output_tensor_index": self.output_tensor_index,
            "normalize_features": self.normalize_features,
            "lazy_loading": self.lazy_loading,
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "thread_safe": self.thread_safe,
            # Cannot serialize preprocessor functions
            "n_preprocessors": len(self.preprocessors),
            # Inference statistics
            "metrics": {
                "total_inference_time": self._total_inference_time,
                "inference_count": self._inference_count,
                "avg_inference_time": self._total_inference_time / max(1, self._inference_count)
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the feature extractor's state.
        
        Args:
            state: Previously saved state dictionary
        """
        # Save old values that we might need to check if a reload is necessary
        old_model_path = self.model_path
        old_feature_layer_index = self.feature_layer_index
        old_input_tensor_index = self.input_tensor_index
        old_output_tensor_index = self.output_tensor_index
        
        # Update state
        self.model_path = state.get("model_path", self.model_path)
        self.feature_layer_index = state.get("feature_layer_index", self.feature_layer_index)
        self.input_tensor_index = state.get("input_tensor_index", self.input_tensor_index)
        self.output_tensor_index = state.get("output_tensor_index", self.output_tensor_index)
        self.normalize_features = state.get("normalize_features", self.normalize_features)
        self.lazy_loading = state.get("lazy_loading", self.lazy_loading)
        self.input_shape = state.get("input_shape", self.input_shape)
        self.input_dtype = state.get("input_dtype", self.input_dtype)
        self.thread_safe = state.get("thread_safe", self.thread_safe)
        
        # Check if we need to reload the interpreter
        need_reload = (
            old_model_path != self.model_path or
            old_feature_layer_index != self.feature_layer_index or
            old_input_tensor_index != self.input_tensor_index or
            old_output_tensor_index != self.output_tensor_index
        )
        
        # Reset the interpreter if needed
        if need_reload and self._interpreter is not None:
            # Thread safety for resource cleanup
            if self.thread_safe:
                with self._lock:
                    self._interpreter = None
                    self._input_details = None
                    self._output_details = None
            else:
                self._interpreter = None
                self._input_details = None
                self._output_details = None
            
            # Preload if not using lazy loading
            if not self.lazy_loading:
                self._load_interpreter()
    
    def add_preprocessor(self, preprocessor: Callable) -> None:
        """Add a preprocessing function to the pipeline.
        
        Args:
            preprocessor: Function to add to the preprocessing pipeline
        """
        self.preprocessors.append(preprocessor)
        
    def clear_preprocessors(self) -> None:
        """Clear all preprocessing functions."""
        self.preprocessors = []
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this feature extractor.
        
        Returns:
            Dictionary containing performance metrics:
            - total_inference_time: Total time spent in inference
            - inference_count: Number of inferences performed
            - avg_inference_time: Average time per inference
        """
        return {
            "total_inference_time": self._total_inference_time,
            "inference_count": self._inference_count,
            "avg_inference_time": 
                self._total_inference_time / max(1, self._inference_count)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information:
            - model_path: Path to the model file
            - input_details: Information about model inputs
            - output_details: Information about model outputs
            - is_loaded: Whether the model is currently loaded
        """
        if not self._interpreter:
            return {
                "model_path": self.model_path,
                "is_loaded": False
            }
        
        # Get information about the model
        return {
            "model_path": self.model_path,
            "input_details": self._input_details,
            "output_details": self._output_details,
            "is_loaded": True
        }
        
    def __del__(self):
        """Clean up resources."""
        # Explicitly delete the interpreter to free resources
        if hasattr(self, '_interpreter') and self._interpreter is not None:
            del self._interpreter
