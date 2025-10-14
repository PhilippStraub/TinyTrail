"""
KNN distance-based drift detection using Page-Hinkley test.

Monitors k-nearest neighbor distances to detect distribution drift
and novel/unknown classes. Uses statistical change detection with
adaptive thresholds and hysteresis for robust on-device monitoring.
"""

import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KNNDistanceMonitor:
    """Drift detector based on k-nearest neighbor distance monitoring.
    
    Detects distribution drift by tracking changes in the distances to nearest
    neighbors. Unknown classes and outliers typically exhibit larger distances
    than known samples. Uses the Page-Hinkley test to identify statistically
    significant increases in average neighbor distances.
    
    Particularly effective for:
    - Detecting novel/unknown object classes
    - Identifying distribution shifts in feature space
    - Real-time anomaly detection with low computational overhead
    """
    
    def __init__(
        self,
        delta: float = 0.1,
        lambda_threshold: float = 5.0,
        exit_threshold_factor: float = 0.7,
        high_confidence_threshold: float = 0.9,
        stable_known_classes: List[str] = None,
        warm_up_samples: int = 5,
        reference_update_interval: int = 50,
        reference_update_factor: float = 0.05,
        pause_reference_update_during_drift: bool = True,
        drift_cooldown_period: int = 10,
        initial_reference_mean: float = None,
        initial_reference_std: float = None,
        reference_stats_path: str = None,
        use_adaptive_thresholds: bool = False,
        allow_runtime_threshold_adaptation: bool = False,
        adaptive_delta_std_multiplier: float = 1.5,
        adaptive_lambda_std_multiplier: float = 5.0,
        aggregation_mode: str = 'mean',  # {'min','mean','median','q'}
        aggregation_q: float = 0.5,
        two_sided: bool = True,
    ):
        """Initialize the KNN distance monitor.
        
        Args:
            delta: Magnitude parameter to allow for small fluctuations (δ)
            lambda_threshold: Threshold for drift detection (λ)
            exit_threshold_factor: Factor to multiply lambda_threshold when determining threshold for exiting drift state
            high_confidence_threshold: Confidence threshold for identifying stable predictions
            stable_known_classes: List of class names considered stable (to skip/reset drift detection when predicted with high confidence)
            warm_up_samples: Number of samples to collect during warm-up
            reference_update_interval: Number of samples between reference updates
            reference_update_factor: Factor for updating reference (β)
            pause_reference_update_during_drift: Whether to pause updating during detected drift
            drift_cooldown_period: Number of samples to wait before triggering another drift event
            initial_reference_mean: Pre-computed reference mean to use instead of warm-up calculation
            initial_reference_std: Pre-computed reference standard deviation for adaptive thresholds
            reference_stats_path: Path to JSON file containing pre-computed reference statistics
            use_adaptive_thresholds: Whether to use reference_std to set delta and lambda thresholds 
            allow_runtime_threshold_adaptation: Whether to update thresholds during runtime (vs. only during initialization)
            adaptive_delta_std_multiplier: Multiplier for reference_std to calculate delta threshold
            adaptive_lambda_std_multiplier: Multiplier for reference_std to calculate lambda threshold
            aggregation_mode: Aggregation mode for k-NN distances ('min','mean','median','q')
            aggregation_q: Quantile value when aggregation_mode='q'
            two_sided: Whether to use two-sided Page-Hinkley test (detects both increases and decreases)
        """
        # Base streaming detection state (inlined from previous base class)
        self.warm_up_samples = warm_up_samples
        self.reference_update_interval = reference_update_interval
        self.reference_update_factor = reference_update_factor
        self.pause_reference_update_during_drift = pause_reference_update_during_drift
        self.drift_cooldown_period = drift_cooldown_period

        self.in_warm_up_phase = True
        self.samples_processed = 0
        self.drift_detected = False
        self.samples_since_last_update = 0
        self.samples_since_last_drift = 0
        self.in_cooldown_period = False
        self.last_drift_timestamp = 0.0
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []

        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.exit_threshold_factor = exit_threshold_factor
        self.high_confidence_threshold = high_confidence_threshold
        self.stable_known_classes = stable_known_classes or []
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.allow_runtime_threshold_adaptation = allow_runtime_threshold_adaptation
        self.adaptive_delta_std_multiplier = adaptive_delta_std_multiplier
        self.adaptive_lambda_std_multiplier = adaptive_lambda_std_multiplier
        self.aggregation_mode = aggregation_mode
        self.aggregation_q = aggregation_q
        self.two_sided = two_sided
        
        # Store pre-computed reference statistics values
        self.initial_reference_mean = initial_reference_mean
        self.initial_reference_std = initial_reference_std
        self.reference_stats_path = reference_stats_path
        
        # PH statistics
        self.reference_mean = None  # μ_ref
        self.reference_std = None   # σ_ref (for adaptive thresholds)
        self.class_stats = {}       # Class-specific statistics
        self.cumulative_sum = 0.0   # m_t
        self.minimum_sum = 0.0      # M_t
        # Two-sided tracking
        self.cum_up = 0.0
        self.cum_down = 0.0
        self.max_up = 0.0
        self.max_down = 0.0
        
        self.last_distances = []
        self.distance_history = deque(maxlen=max(500, warm_up_samples * 5))
        
        # Initialize reference_mean from pre-computed statistics if provided
        if self.initial_reference_mean is not None:
            self.reference_mean = self.initial_reference_mean
            self.in_warm_up_phase = False
            self.warm_up_values = None
            logger.info(f"Using pre-computed reference mean: {self.reference_mean}")
        elif self.reference_stats_path:
            try:
                self._load_reference_statistics()
            except Exception as e:
                logger.warning(f"Failed to load reference statistics from {self.reference_stats_path}: {e}")
                self.warm_up_values = [] if self.in_warm_up_phase else None
        else:
            self.warm_up_values = [] if self.in_warm_up_phase else None
        
        logger.debug(
            f"Initialized KNNDistanceMonitor with lambda_threshold={lambda_threshold}, "
            f"delta={delta}, warm_up_samples={warm_up_samples}, "
            f"use_precomputed_stats={self.reference_mean is not None}"
        )
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.callbacks.append(callback)

    def _notify_callbacks(self, drift_info: Dict[str, Any]) -> bool:
        if "timestamp" not in drift_info:
            drift_info["timestamp"] = time.time()
        if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
            drift_info["in_cooldown_period"] = True
            drift_info["samples_since_last_drift"] = self.samples_since_last_drift
            drift_info["drift_cooldown_period"] = self.drift_cooldown_period
            return False
        self.in_cooldown_period = True
        self.samples_since_last_drift = 0
        self.last_drift_timestamp = drift_info.get("timestamp", 0.0) or 0.0
        for cb in self.callbacks:
            try:
                try:
                    cb(drift_info)
                except TypeError:
                    cb(drift_info, {})
            except Exception as e:
                logger.error(f"Error in drift callback: {e}")
        return True

    def should_update_reference(self, prediction=None, confidence=None) -> bool:
        """Only update reference with stable, high-confidence predictions.
        
        Args:
            prediction: Current predicted class
            confidence: Confidence score of the prediction
            
        Returns:
            bool: Whether reference statistics should be updated
        """
        if self.in_warm_up_phase:
            return False
        if self.pause_reference_update_during_drift and self.drift_detected:
            return False
        if self.samples_since_last_update < self.reference_update_interval:
            return False
        
        # GATING: Only update with stable, high-confidence predictions
        if self.stable_known_classes and prediction is not None:
            # Must be a stable known class
            if prediction not in self.stable_known_classes:
                return False
            # Must have high confidence
            if confidence is None or confidence < self.high_confidence_threshold:
                return False
        
        return True

    def _process_sample(self, _: Any) -> None:
        self.samples_processed += 1
        self.samples_since_last_update += 1
        if self.in_cooldown_period:
            self.samples_since_last_drift += 1
            if self.samples_since_last_drift >= self.drift_cooldown_period:
                self.in_cooldown_period = False
        if self.in_warm_up_phase and self.samples_processed >= self.warm_up_samples:
            self.in_warm_up_phase = False

    def update(self, record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Update the detector with new data.
        
        Args:
            record: Dictionary containing prediction data. Should have either:
                  - '_knn_distances' key with a list of distances to nearest neighbors
                  - 'classifier' key with an object having '_last_distances' attribute
                  - 'confidence' key which can be used as a fallback
            
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if record is None:
            logger.warning("Record is None, skipping update")
            return False, None
        
        if not isinstance(record, dict):
            logger.warning(f"Record is not a dictionary, received: {type(record).__name__}, skipping update")
            return False, None
        
        prediction = record.get('prediction')
        confidence = record.get('confidence')
        #Philipp edit
        logger.debug(f"Confidence-Reset-Check: prediction={prediction} ({type(prediction)}), "
             f"stable_known_classes={self.stable_known_classes}, "
             f"confidence={confidence} ({type(confidence)}), "
             f"high_confidence_threshold={self.high_confidence_threshold}, "
             f"drift_detected={self.drift_detected}")
        
        if prediction and confidence is not None and self.stable_known_classes:
            if confidence >= self.high_confidence_threshold and prediction in self.stable_known_classes:
                # Only reset if ALREADY in drift state (helps exit drift quickly)
                if self.drift_detected:
                    logger.info(f"KNNDistanceMonitor: Reset due to high-confidence ({confidence:.2f}) known class '{prediction}'")
                    self.reset_ph_stats()
                    return False, None
                # IMPORTANT: No early return! Continue with normal PH processing
            
        try:    
            self._process_sample(record)
            if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
                return False, None
            
            distances = self._extract_distances(record)
            if distances is None or len(distances) == 0:
                logger.warning(f"Could not extract valid distances from record with keys: {list(record.keys())}. Skipping update for this sample.")
                return False, None
        except Exception as e:
            logger.warning(f"Error processing record in KNNDistanceMonitor: {str(e)}")
            return False, None
        
        # Aggregate k-distances
        self.last_distances = distances
        d = np.array(distances, dtype=np.float32)
        mode = (self.aggregation_mode or 'mean').lower()
        if mode == 'min':
            agg_distance = float(np.min(d))
        elif mode == 'median':
            agg_distance = float(np.median(d))
        elif mode == 'q':
            q = float(self.aggregation_q)
            q = min(max(q, 0.0), 1.0)
            agg_distance = float(np.quantile(d, q))
        else:
            agg_distance = float(np.mean(d))
        self.distance_history.append(agg_distance)
        
        # Extreme outlier: x > μ_ref + 3*σ_ref (more reasonable than 3*μ_ref)
        if self.reference_mean is not None:
            # Use std-based threshold if available, otherwise fall back to factor-based
            if self.reference_std and self.reference_std > 0:
                outlier_threshold = self.reference_mean + 3.0 * self.reference_std
                if agg_distance > outlier_threshold:
                    std_deviations = (agg_distance - self.reference_mean) / self.reference_std
                    drift_info = {
                        'detector_type': 'KNNDistanceMonitor',
                        'metric': 'neighbor_distance',
                        'current_value': agg_distance,
                        'reference_mean': self.reference_mean,
                        'reference_std': self.reference_std,
                        'immediate_outlier': True,
                        'outlier_threshold': outlier_threshold,
                        'std_deviations': std_deviations,
                        'timestamp': time.time()
                    }
                    logger.warning(f"KNNDistanceMonitor: Immediate outlier detected! Distance={agg_distance:.4f}, "
                                 f"Reference={self.reference_mean:.4f}±{self.reference_std:.4f}, "
                                 f"Deviations={std_deviations:.2f}σ")
                    self._notify_callbacks(drift_info)
                    return True, drift_info
            elif agg_distance > self.reference_mean * 3:
                # Fallback to original factor-based detection if no std available
                drift_info = {
                    'detector_type': 'KNNDistanceMonitor',
                    'metric': 'neighbor_distance',
                    'current_value': agg_distance,
                    'reference_mean': self.reference_mean,
                    'reference_std': self.reference_std,
                    'immediate_outlier': True,
                    'outlier_factor': agg_distance / self.reference_mean,
                    'timestamp': time.time()
                }
                logger.warning(f"KNNDistanceMonitor: Immediate outlier detected! Distance={agg_distance:.4f}, "
                             f"Reference={self.reference_mean:.4f}, Factor={agg_distance/self.reference_mean:.2f}x")
                self._notify_callbacks(drift_info)
                return True, drift_info
        
        if len(self.distance_history) % 10 == 0:
            logger.debug(f"KNNDistanceMonitor: Agg distance = {agg_distance:.4f} (mode={mode})")
        
        drift_info = None
        
        # Handle warm-up phase
        if self.in_warm_up_phase:
            self.warm_up_values.append(agg_distance)
            return False, None
        elif self.reference_mean is None and not self.in_warm_up_phase:
            self.reference_mean = np.mean(self.warm_up_values) if len(self.warm_up_values) > 0 else agg_distance
            
            if len(self.warm_up_values) > 1:
                self.reference_std = np.std(self.warm_up_values)
                logger.info(f"KNNDistanceMonitor: Initialized reference mean={self.reference_mean:.4f}, std={self.reference_std:.4f}")
            else:
                self.reference_std = self.reference_mean * 0.1
                logger.info(f"KNNDistanceMonitor: Initialized reference mean={self.reference_mean:.4f}, using default std={self.reference_std:.4f}")
            
            if self.use_adaptive_thresholds and self.reference_std > 0:
                original_delta = self.delta
                original_lambda = self.lambda_threshold
                
                # Set thresholds based on reference_std
                self.delta = self.reference_std * self.adaptive_delta_std_multiplier
                self.lambda_threshold = self.reference_std * self.adaptive_lambda_std_multiplier
                
                logger.info(f"KNNDistanceMonitor: Using adaptive thresholds after warm-up: "
                          f"delta={original_delta:.4f}->{self.delta:.4f}, "
                          f"lambda={original_lambda:.4f}->{self.lambda_threshold:.4f}")
            
            self.warm_up_values = None  # Free memory
            self.cumulative_sum = 0.0
            self.minimum_sum = 0.0
            return False, None
        
        newly_detected_drift_by_ph = False
        drift_direction = None
        ph_value = 0.0
        
        try:
            # Use class-specific reference if available and applicable
            mu = self.reference_mean if self.reference_mean is not None else 0.0
            sigma = self.reference_std
            ref_source = "global"
            
            # Only use class-specific stats for HIGH-CONFIDENCE predictions (Gate!)
            if (self.class_stats and prediction
                and prediction in self.stable_known_classes
                and prediction in self.class_stats
                and confidence is not None
                and confidence >= self.high_confidence_threshold):
                cs = self.class_stats[prediction]
                mu = float(cs.get('mean', mu))
                sigma = float(cs.get('std', sigma)) if 'std' in cs else sigma
                ref_source = f"class:{prediction}"
                logger.debug(f"Using class-specific stats for '{prediction}' (conf={confidence:.2f}): mean={mu:.4f}, std={sigma}")
            else:
                if prediction and confidence is not None:
                    logger.debug(f"Using GLOBAL stats for PH (conf={confidence:.2f})")
            
            # Error relative to reference
            err = agg_distance - mu
            
            if self.two_sided:
                # Two-sided Page-Hinkley test
                # Track upward drift (distance increases)
                self.cum_up = max(0.0, self.cum_up + (err - self.delta))
                self.max_up = max(self.max_up, self.cum_up)
                drift_up = self.cum_up > self.lambda_threshold
                
                # Track downward drift (distance decreases)
                self.cum_down = max(0.0, self.cum_down + (-err - self.delta))
                self.max_down = max(self.max_down, self.cum_down)
                drift_down = self.cum_down > self.lambda_threshold
                
                was_drift_detected = self.drift_detected
                exit_drift_threshold = self.lambda_threshold * self.exit_threshold_factor
                
                if not self.drift_detected:
                    if drift_up or drift_down:
                        self.drift_detected = True
                        newly_detected_drift_by_ph = True
                        drift_direction = "down" if drift_down and (self.cum_down >= self.cum_up) else "up"
                        ph_value = self.cum_down if drift_direction == "down" else self.cum_up
                        
                        logger.info(f"KNNDistanceMonitor: Entering drift state ({drift_direction}). "
                                  f"cum_{drift_direction}={ph_value:.2f} > threshold {self.lambda_threshold:.2f}")
                        logger.info(f"KNNDistanceMonitor: agg_distance={agg_distance:.2f}, reference_mean={mu:.2f}, "
                                  f"err={err:.2f}, cum_up={self.cum_up:.2f}, cum_down={self.cum_down:.2f}")
                        
                        if 'prediction' in record and 'confidence' in record:
                            logger.info(f"KNNDistanceMonitor: Current prediction='{record['prediction']}', confidence={record['confidence']:.2f}")
                else:
                    # Check if we should exit drift state
                    if self.cum_up <= exit_drift_threshold and self.cum_down <= exit_drift_threshold:
                        self.drift_detected = False
                        logger.info(f"KNNDistanceMonitor: Exited drift state. Both cum_up={self.cum_up:.2f} and "
                                  f"cum_down={self.cum_down:.2f} <= exit_threshold {exit_drift_threshold:.2f}")
                        self.reset_ph_stats()
            else:
                # Original single-sided Page-Hinkley test (backward compatibility)
                # Page-Hinkley: ε_t = x_t - (μ_ref + δ)
                deviation = agg_distance - (mu + self.delta)
                # Cumulative: m_t = m_{t-1} + ε_t
                self.cumulative_sum += deviation
                # Minimum: M_t = min(M_{t-1}, m_t)
                self.minimum_sum = min(self.minimum_sum, self.cumulative_sum)
                # Test statistic: PH_t = m_t - M_t
                ph_value = self.cumulative_sum - self.minimum_sum
                
                was_drift_detected = self.drift_detected
                exit_drift_threshold = self.lambda_threshold * self.exit_threshold_factor
                
                if not self.drift_detected:
                    if ph_value > self.lambda_threshold:
                        self.drift_detected = True
                        newly_detected_drift_by_ph = True
                        drift_direction = "up"
                        logger.info(f"KNNDistanceMonitor: Entering drift state. PH value {ph_value:.2f} > threshold {self.lambda_threshold:.2f}")
                        logger.info(f"KNNDistanceMonitor: agg_distance={agg_distance:.2f}, reference_mean={mu:.2f}, " +
                                  f"deviation={deviation:.2f}, cumulative_sum={self.cumulative_sum:.2f}")
                        
                        if 'prediction' in record and 'confidence' in record:
                            logger.info(f"KNNDistanceMonitor: Current prediction='{record['prediction']}', confidence={record['confidence']:.2f}")
                else:
                    if ph_value <= exit_drift_threshold:
                        self.drift_detected = False
                        logger.info(f"KNNDistanceMonitor: Exited drift state. PH value {ph_value:.2f} <= exit_threshold {exit_drift_threshold:.2f}")
                        self.reset_ph_stats()
            
            if self.drift_detected and not was_drift_detected:
                std_info = ""
                if self.reference_std:
                    if self.reference_mean > 0 and self.reference_std > 0:
                        std_distance = (agg_distance - self.reference_mean) / self.reference_std
                        std_info = f", deviation: {std_distance:.2f} standard deviations"
                
                logger.warning(f"KNNDistanceMonitor: DRIFT DETECTED by Page-Hinkley test! " +
                             f"PH value: {ph_value:.2f} > threshold: {self.lambda_threshold:.2f}, " +
                             f"avg_distance: {agg_distance:.2f}, reference: {self.reference_mean:.2f}, " +
                             f"ratio: {(agg_distance/max(0.01, self.reference_mean)):.2f}x" + std_info)
            
            if self.samples_processed % 10 == 0:
                logger.debug(f"KNNDistanceMonitor PH value: {ph_value:.2f}, threshold: {self.lambda_threshold:.2f}, " +
                            f"agg_distance: {agg_distance:.2f}, reference: {self.reference_mean:.2f}")
        except Exception as e:
            logger.warning(f"Error in Page-Hinkley calculation: {str(e)}")
            was_drift_detected = self.drift_detected
            self.drift_detected = False
        
        is_unknown_object = False
        
        if self.in_cooldown_period and self.samples_since_last_drift < self.drift_cooldown_period:
            self.drift_detected = False
            is_unknown_object = False
            logger.debug(f"In cooldown period ({self.samples_since_last_drift}/{self.drift_cooldown_period}) - suppressing drift detection")
        
        if newly_detected_drift_by_ph:
            drift_info = {
                'detector_type': 'KNNDistanceMonitor',
                'metric': 'neighbor_distance',
                'current_value': agg_distance,
                'reference_mean': self.reference_mean,
                'reference_std': self.reference_std,
                'ph_value': ph_value if 'ph_value' in locals() else 0.0,
                'threshold': self.lambda_threshold,
                'distances': distances,
                'timestamp': time.time()
            }
            
            # Add two-sided specific information
            if self.two_sided:
                drift_info['direction'] = drift_direction
                drift_info['cum_up'] = self.cum_up
                drift_info['cum_down'] = self.cum_down
                drift_info['delta'] = self.delta
            
            if self.reference_std and self.reference_std > 0:
                std_distance = (agg_distance - self.reference_mean) / self.reference_std
                drift_info['std_distance'] = std_distance
            
            prediction_info = ""
            if 'prediction' in record:
                prediction_info = f", Prediction={record['prediction']}"
                
            logger.info(
                f"KNNDistanceMonitor: Drift detected! Agg distance={agg_distance:.4f}, "
                f"Reference={self.reference_mean:.4f}{prediction_info}, " +
                (f"PH value={ph_value:.4f}" if 'ph_value' in locals() else "Direct distance-based detection")
            )
            
            self._notify_callbacks(drift_info)
        
        if self.should_update_reference(prediction, confidence):
            # EWMA: μ = (1-β)*μ + β*x
            beta = self.reference_update_factor
            self.reference_mean = (1 - beta) * self.reference_mean + beta * agg_distance
            
            if self.reference_std is not None:
                recent_distances = list(self.distance_history)[-30:]
                if recent_distances and len(recent_distances) > 5:
                    current_std = max(float(np.std(recent_distances)), 1e-6)
                    self.reference_std = (1 - beta) * self.reference_std + beta * current_std
                    
                    if self.use_adaptive_thresholds and self.allow_runtime_threshold_adaptation:
                        original_delta = self.delta
                        original_lambda = self.lambda_threshold
                        
                        self.delta = self.reference_std * self.adaptive_delta_std_multiplier
                        self.lambda_threshold = self.reference_std * self.adaptive_lambda_std_multiplier
                        
                        if (abs(original_delta - self.delta) > 0.1 * original_delta or 
                            abs(original_lambda - self.lambda_threshold) > 0.1 * original_lambda):
                            logger.info(f"KNNDistanceMonitor: Updated adaptive thresholds during runtime: "
                                      f"delta={original_delta:.4f}->{self.delta:.4f}, "
                                      f"lambda={original_lambda:.4f}->{self.lambda_threshold:.4f}")
            
            self.samples_since_last_update = 0
            if logger.isEnabledFor(logging.DEBUG):
                std_value = f"{self.reference_std:.4f}" if self.reference_std is not None else "None"
                logger.debug(f"KNNDistanceMonitor: Updated reference mean to {self.reference_mean:.4f}, std to {std_value}")
        
        return self.drift_detected, drift_info
    
    def _extract_distances(self, record: Dict[str, Any]) -> Optional[List[float]]:
        """Extract distance information from the record.
        
        Tries multiple sources in order of preference:
        1. '_knn_distances' key in the record
        2. 'classifier' object with '_last_distances' attribute
        3. '_last_distances' key in the record
        
        Args:
            record: Dictionary with prediction data
        
        Returns:
            List of distances or None if not found
        """
        try:
            # Case 1: Direct distances in record
            if '_knn_distances' in record and isinstance(record['_knn_distances'], (list, tuple, np.ndarray)):
                return list(record['_knn_distances'])
            
            # Case 2: Classifier object with distances attribute
            if 'classifier' in record and record['classifier'] is not None:
                if hasattr(record['classifier'], '_last_distances'):
                    distances = record['classifier']._last_distances
                    if distances is not None:
                        return list(distances)
                # Some classifiers store distances in a different attribute
                if hasattr(record['classifier'], 'last_distances'):
                    distances = record['classifier'].last_distances
                    if distances is not None:
                        return list(distances)
            
            # Case 3: Last distances in record
            if '_last_distances' in record and isinstance(record['_last_distances'], (list, tuple, np.ndarray)):
                return list(record['_last_distances'])
            
            # Case 4: Extract from raw result if available
            if 'raw_result' in record and isinstance(record['raw_result'], dict):
                raw = record['raw_result']
                if 'distances' in raw and isinstance(raw['distances'], (list, tuple, np.ndarray)):
                    return list(raw['distances'])
            
            # Case 5: Try to extract from metadata if present
            if 'metadata' in record and isinstance(record['metadata'], dict):
                meta = record['metadata']
                if 'distances' in meta and isinstance(meta['distances'], (list, tuple, np.ndarray)):
                    return list(meta['distances'])
                if 'neighbor_distances' in meta and isinstance(meta['neighbor_distances'], (list, tuple, np.ndarray)):
                    return list(meta['neighbor_distances'])
            
            # These fallbacks have been removed since we shouldn't hardcode values
            # and instead rely on the actual distances computed by the classifier
            
            # Log record keys to help debug
            logger.debug(f"KNNDistanceMonitor could not extract distances. Record keys: {list(record.keys())}")
            
            # Return None to indicate that no valid distances were found
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting distances from record: {str(e)}")
            return None  # Return None to indicate extraction error
    
    def check_for_drift(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if drift has been detected.
        
        Returns:
            Tuple of (drift_detected, drift_info)
        """
        if not self.drift_detected:
            return False, None
        
        # Calculate average distance if available
        avg_distance = None
        if self.last_distances:
            avg_distance = sum(self.last_distances) / len(self.last_distances)
        
        drift_info = {
            'detector_type': 'KNNDistanceMonitor',
            'metric': 'neighbor_distance',
            'current_value': avg_distance,
            'reference_mean': self.reference_mean,
            'reference_std': self.reference_std,  # Include reference_std in check_for_drift too
            'ph_value': self.cumulative_sum - self.minimum_sum,
            'threshold': self.lambda_threshold,
            'std_distance': (avg_distance - self.reference_mean) / self.reference_std if self.reference_std and self.reference_std > 0 else None
        }
        
        return True, drift_info
    
    def reset_ph_stats(self) -> None:
        """Reset only the Page-Hinkley test statistics.
        
        This resets the drift flag and Page-Hinkley statistics but keeps
        the reference statistics for continuous monitoring.
        """
        self.drift_detected = False
        self.cumulative_sum = 0.0
        self.minimum_sum = 0.0
        # Reset two-sided tracking variables
        self.cum_up = 0.0
        self.cum_down = 0.0
        self.max_up = 0.0
        self.max_down = 0.0
        # FIX: Clear distance history to prevent stale values after reset
        self.distance_history.clear()
        
        logger.debug("KNNDistanceMonitor: Reset Page-Hinkley statistics and cleared distance history")
    
    def reset(self) -> None:
        """Reset the detector state completely (keeps reference stats)."""
        self.drift_detected = False
        self.samples_since_last_update = 0
        self.in_cooldown_period = False
        self.samples_since_last_drift = 0
        self.last_drift_timestamp = 0.0
        self.reset_ph_stats()
        logger.debug("KNNDistanceMonitor: Reset detector state completely")
    
    def _load_reference_statistics(self) -> None:
        """Load pre-computed reference statistics from a JSON file.
        
        This method loads reference statistics (mean, std, etc.) from a JSON file
        specified by self.reference_stats_path.
        """
        if not self.reference_stats_path:
            logger.warning("No reference statistics path provided")
            return
            
        import json
        import os
        
        if not os.path.exists(self.reference_stats_path):
            logger.warning(f"Reference statistics file not found: {self.reference_stats_path}")
            return
            
        try:
            with open(self.reference_stats_path, 'r') as f:
                stats = json.load(f)
                
            # Extract relevant statistics
            if 'reference_mean' in stats:
                self.reference_mean = float(stats['reference_mean'])
                logger.info(f"Loaded reference mean: {self.reference_mean}")
                
                # Load class-specific statistics if available
                if 'class_statistics' in stats:
                    self.class_stats = stats['class_statistics']
                    logger.info(f"Loaded class-specific statistics for {len(self.class_stats)} classes")
                
                # Load standard deviation for adaptive thresholds
                adjusted_message = ""  # Initialize before conditional blocks
                if 'reference_std' in stats:
                    self.reference_std = float(stats['reference_std'])
                    self.initial_reference_std = self.reference_std
                    logger.info(f"Loaded reference std: {self.reference_std}")
                    
                    # Adjust thresholds based on loaded statistics if they were provided in config
                    original_delta = self.delta
                    original_lambda = self.lambda_threshold
                    
                    # Only adjust if explicitly using loaded statistics (keeping configured values as is)
                    # This allows for explicit configuration but also provides sensible defaults
                    if self.reference_std > 0:
                        # Use reference_std to validate if the current thresholds make sense
                        suggested_delta = self.reference_std * self.adaptive_delta_std_multiplier
                        suggested_lambda = self.reference_std * self.adaptive_lambda_std_multiplier
                        
                        # Log comparison between configured and suggested values
                        logger.info(f"Current delta: {self.delta}, Suggested delta: {suggested_delta}")
                        logger.info(f"Current lambda: {self.lambda_threshold}, Suggested lambda: {suggested_lambda}")
                        
                        # If adaptive thresholds are enabled, update parameters
                        if self.use_adaptive_thresholds:
                            self.delta = suggested_delta
                            self.lambda_threshold = suggested_lambda
                            adjusted_message = f" (using adaptive thresholds: delta={suggested_delta:.2f}, lambda={suggested_lambda:.2f})"
                            logger.info(f"Using adaptive thresholds based on reference statistics")
                        else:
                            adjusted_message = " (kept explicit configuration)"
                    
                # Skip warm-up phase if reference mean is available
                self.in_warm_up_phase = False
                self.warm_up_values = None
                
                logger.info(f"Successfully loaded reference statistics from file{adjusted_message}")
            else:
                logger.warning(f"No reference mean found in statistics file: {self.reference_stats_path}")
        except Exception as e:
            logger.error(f"Error loading reference statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the detector.
        
        Returns:
            State dictionary
        """
        state = {
            "warm_up_samples": self.warm_up_samples,
            "reference_update_interval": self.reference_update_interval,
            "reference_update_factor": self.reference_update_factor,
            "pause_reference_update_during_drift": self.pause_reference_update_during_drift,
            "drift_cooldown_period": self.drift_cooldown_period,
            "in_warm_up_phase": self.in_warm_up_phase,
            "samples_processed": self.samples_processed,
            "drift_detected": self.drift_detected,
            "samples_since_last_update": self.samples_since_last_update,
            "in_cooldown_period": self.in_cooldown_period,
            "samples_since_last_drift": self.samples_since_last_drift,
            "last_drift_timestamp": self.last_drift_timestamp,
        }
        state.update({
            'reference_mean': self.reference_mean,
            'reference_std': self.reference_std,  # Added reference_std to state
            'cumulative_sum': self.cumulative_sum,
            'minimum_sum': self.minimum_sum,
            'lambda_threshold': self.lambda_threshold,
            'exit_threshold_factor': self.exit_threshold_factor,
            'high_confidence_threshold': self.high_confidence_threshold,
            'stable_known_classes': self.stable_known_classes,
            'delta': self.delta,
            'initial_reference_mean': self.initial_reference_mean,
            'initial_reference_std': self.initial_reference_std,
            'reference_stats_path': self.reference_stats_path,
            'use_adaptive_thresholds': self.use_adaptive_thresholds,
            'allow_runtime_threshold_adaptation': self.allow_runtime_threshold_adaptation,
            'adaptive_delta_std_multiplier': self.adaptive_delta_std_multiplier,
            'adaptive_lambda_std_multiplier': self.adaptive_lambda_std_multiplier,
            'aggregation_mode': self.aggregation_mode,
            'aggregation_q': self.aggregation_q,
            'warm_up_values': self.warm_up_values if self.warm_up_values is not None else [],
            'last_distances': self.last_distances,
            'recent_distance_history': list(self.distance_history)[-100:] if self.distance_history else []
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the detector state.
        
        Args:
            state: Previously saved state dictionary
        """
        # Base state restoration
        self.warm_up_samples = state.get("warm_up_samples", self.warm_up_samples)
        self.reference_update_interval = state.get("reference_update_interval", self.reference_update_interval)
        self.reference_update_factor = state.get("reference_update_factor", self.reference_update_factor)
        self.pause_reference_update_during_drift = state.get("pause_reference_update_during_drift", self.pause_reference_update_during_drift)
        self.drift_cooldown_period = state.get("drift_cooldown_period", self.drift_cooldown_period)
        self.in_warm_up_phase = state.get("in_warm_up_phase", self.in_warm_up_phase)
        self.samples_processed = state.get("samples_processed", self.samples_processed)
        self.drift_detected = state.get("drift_detected", self.drift_detected)
        self.samples_since_last_update = state.get("samples_since_last_update", self.samples_since_last_update)
        self.in_cooldown_period = state.get("in_cooldown_period", self.in_cooldown_period)
        self.samples_since_last_drift = state.get("samples_since_last_drift", self.samples_since_last_drift)
        self.last_drift_timestamp = state.get("last_drift_timestamp", self.last_drift_timestamp)
        self.reference_mean = state.get('reference_mean', self.reference_mean)
        self.reference_std = state.get('reference_std', self.reference_std)  # Restore reference_std
        self.cumulative_sum = state.get('cumulative_sum', self.cumulative_sum)
        self.minimum_sum = state.get('minimum_sum', self.minimum_sum)
        self.lambda_threshold = state.get('lambda_threshold', self.lambda_threshold)
        self.exit_threshold_factor = state.get('exit_threshold_factor', self.exit_threshold_factor)
        self.high_confidence_threshold = state.get('high_confidence_threshold', self.high_confidence_threshold)
        self.stable_known_classes = state.get('stable_known_classes', self.stable_known_classes)
        self.delta = state.get('delta', self.delta)
        self.initial_reference_mean = state.get('initial_reference_mean', self.initial_reference_mean)
        self.initial_reference_std = state.get('initial_reference_std', self.initial_reference_std)
        self.reference_stats_path = state.get('reference_stats_path', self.reference_stats_path)
        self.use_adaptive_thresholds = state.get('use_adaptive_thresholds', self.use_adaptive_thresholds)
        self.allow_runtime_threshold_adaptation = state.get('allow_runtime_threshold_adaptation', self.allow_runtime_threshold_adaptation)
        self.adaptive_delta_std_multiplier = state.get('adaptive_delta_std_multiplier', self.adaptive_delta_std_multiplier)
        self.adaptive_lambda_std_multiplier = state.get('adaptive_lambda_std_multiplier', self.adaptive_lambda_std_multiplier)
        self.aggregation_mode = state.get('aggregation_mode', self.aggregation_mode)
        self.aggregation_q = state.get('aggregation_q', self.aggregation_q)
        
        warm_up_values = state.get('warm_up_values', [])
        self.warm_up_values = warm_up_values if self.in_warm_up_phase else None
        
        self.last_distances = state.get('last_distances', [])
        
        # Restore distance history if available
        if 'recent_distance_history' in state:
            from collections import deque
            self.distance_history = deque(state['recent_distance_history'], maxlen=max(500, self.warm_up_samples * 5))
