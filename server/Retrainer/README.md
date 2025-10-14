# On‑Device Drift Detection (Pi Zero 2W)

Real‑time drift monitoring on a Raspberry Pi Zero 2W using a quantized TFLite backbone for feature extraction, a lightweight k‑NN for classification, and a Page‑Hinkley–based KNN distance monitor for drift detection.

---

## Quick Start

### Prerequisites

* Docker (for development environment)
* Python 3.8+ (for Raspberry Pi deployment)
* Camera module (for Pi Zero 2W)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/on-device-drift-detection.git
cd on-device-drift-detection
```

### 2. Start Development Container

```bash
# Open in VS Code
code .
# Then: Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# Or use Docker directly
docker build -t tinylcm-dev .devcontainer/
docker run -it -v $(pwd):/workspace tinylcm-dev
```

### 3. Train Model (in DevContainer)

```bash
# Scenario 1: Train without Lego
cd examples/model_training/scenario_lego_drift
python train_core.py
python make_feature_processor.py

# Scenario 2: Train with Lego
cd examples/model_training/scenario_ball_drift
python train_core.py
python make_feature_processor.py
```

### 4. Create Initial State

```bash
cd examples/python_examples/scenario_lego_drift
python create_initial_state.py
```


---

## Development Environment

### Why Use DevContainer?

The DevContainer ensures dependency parity between development and the Pi target:

* **Python 3.11** – Matches Raspberry Pi OS
* **TensorFlow 2.15.0** – Training and feature export
* **NumPy 1.24.2** – Compatible with Pi Zero

### DevContainer Setup

1. Install **Docker Desktop**
2. Install **VS Code** + **Dev Containers** extension
3. Open project in container:

   ```bash
   # VS Code will prompt to reopen in container
   # Or manually: Ctrl+Shift+P → "Dev Containers: Reopen in Container"
   ```

The container automatically:

* Installs all dependencies from `.devcontainer/requirements.txt`
* Mounts your project directory
* Sets up a Python environment
* Configures Git credentials

---

## High‑Level Architecture

```
Camera → Resize(224×224) → TFLiteFeatureExtractor → [Optional] StandardScaler+PCA →
  → LightweightKNN (predict + probabilities) →
  → KNNDistanceMonitor (Page‑Hinkley on k‑NN distances) → Drift events
                           ↘ JSON/Log (frames, metrics, drift)
```

**Key ideas**

* We do **classification** via a tiny, memory‑friendly **k‑NN** on top of learned features.
* We do **drift detection** by monitoring how far the current feature is from the reference pool of known/stable classes (via k‑NN distances), then running a **Page‑Hinkley** change‑detection test.

---

## The Math (short & practical)

### 1) Feature extraction & quantization

* Input image `x ∈ ℝ^{H×W×3}` is resized to `224×224` and normalized to `[-1, 1]`.
* If the TFLite model is quantized (INT8/UINT8), we apply:

  Quantize:  $q = \operatorname{round}(x/\text{scale} + \text{zero\_point})$

  Dequantize (for outputs):  $y = \text{scale} \cdot (q - \text{zero\_point})$

### 2) Optional Standardization + PCA

* Standardize features:  $z = (y - \mu) / \sigma$
* PCA projection (keep top `d` components):  $f = z \cdot W^\top$
* (Optional) L2‑normalize `f`.

### 3) k‑Nearest Neighbors (classification)

* Distances to each stored sample `x_i`:

  * Euclidean: $d_i = \lVert f - x_i \rVert_2$
  * Cosine: $d_i = 1 - \cos(f, x_i)$
* Pick `k` smallest distances. With **inverse distance weighting (IDW)**:
  $w_i = 1/(\varepsilon + d_i^p)$ and class score is the sum of weights per class.
* Class probability ≈ normalized class scores from the `k` neighbors.

### 4) Aggregated neighbor distance (for drift)

* From the same neighbors, compute a robust aggregate:

  * `mean`, `median`, `min`, or **`q`‑quantile** (we often use `q = 0.9`).
* This single scalar per frame is the **signal** we feed to Page‑Hinkley.

### 5) Page‑Hinkley (PH) change detection

Let $x_t$ be the aggregated distance at time `t`, and $\mu_{ref}$ the reference mean.

* **One‑sided (up)** – trigger when distances **increase** a lot:

  Deviation: $\epsilon_t = x_t - (\mu_{ref} + \delta)$
  Cumulative: $m_t = m_{t-1} + \epsilon_t$
  Minimum: $M_t = \min(M_{t-1}, m_t)$
  **PH**: $PH_t = m_t - M_t$
  Drift if $PH_t > \lambda$.

* **Two‑sided** – detect both **up** and **down** shifts (e.g., distances becoming much smaller or larger than reference):

  Let $e_t = x_t - \mu_{ref}$. Then track

  $\text{cum\_up} = \max(0, \text{cum\_up} + (e_t - \delta))$
  $\text{cum\_down} = \max(0, \text{cum\_down} + (-e_t - \delta))$
  Drift if either exceeds $\lambda$. Direction is `up` or `down`.

* **Thresholds**

  * $\delta$ (delta) filters small fluctuations
  * $\lambda$ (lambda) is the trigger threshold
  * Exit hysteresis: you exit drift when cum values fall below `exit_threshold_factor × λ`.

* **Adaptive thresholds (optional)**
  If a reference std $\sigma_{ref}$ is known:
  $\delta = \alpha \cdot \sigma_{ref}$, $\lambda = \beta \cdot \sigma_{ref}$ (multipliers `α`, `β`).

---

## Components & Responsibilities

### `TFLiteFeatureExtractor`

* Loads a quantized **TFLite** model on the Pi.
* Applies the same pre‑/de‑quantization as at training time.
* Exposes an **intermediate layer** (usually a pooled embedding) as the feature vector.
* Optional **L2 normalization** for cosine metrics.

### `StandardScalerPCATransformer`

* Loads scaler (`μ`, `σ`) and PCA projection matrix `W` from `feature_processor.npz`.
* Applies `z = (x−μ)/σ` then `f = z·Wᵀ`.
* Keeps the runtime identical to training exports.

### `LightweightKNN`

* Stores up to `max_samples` feature vectors with labels and timestamps.
* Predicts with **brute‑force k‑NN** (fast enough for ≤200 samples on Pi Zero).
* Supports **IDW** (inverse distance weighting) with exponent `idw_power`.
* Caches last neighbor distances for logging and drift.

### `KNNDistanceMonitor` (drift)

* Extracts the **k nearest distances** to the **reference pool** (typically *stable* classes only, e.g. `stone`, `leaf`, `coin`).
* Aggregates them (e.g. `q=0.9`).
* Runs **Page‑Hinkley**:

  * **Two‑sided** if you want to flag **both** “distances too big” and “distances too small” vs reference.
  * **One‑sided (up)** if you only care about novel/unknown objects (distances **increase**).
* **High‑confidence reset**: if a known class is predicted with confidence ≥ `high_confidence_threshold`, PH stats reset, avoiding sticky drift.
* **Reference update (EWMA)**: `μ_ref ← (1−β)·μ_ref + β·x_t` every `reference_update_interval` frames (paused during drift if configured).

---

## Reference Statistics (Leave‑One‑Out)

We build a robust baseline from the **stored k‑NN samples** (usually **only stable classes**):

1. For each sample `i`, compute distances to all **other** samples (exclude itself).
2. Take the `k` nearest distances; aggregate with the same mode (`mean` / `q`‑quantile).
3. Collect the values over all `i` to estimate `reference_mean` and `reference_std`.
4. Save to `knn_reference_stats.json` and load at runtime.

This ensures **identical scaling** and aggregation to what the monitor uses online.

---

## Config Reference (`config.json`)

Below we document the important fields, what they do, and what happens if you change them. (Names follow your provided `config.json`.)

### `device`

* `name`, `description` – Metadata only.
* `log_level` – Logging verbosity (`INFO`, `DEBUG`). Higher verbosity helps debugging but is noisier.

### `camera`

* `resolution`: `[640, 480]` – Capture resolution from the camera. Larger = slower but potentially clearer features.
* `framerate`: `10` – Target FPS.
* `rotation`: `0` – Degrees to rotate the image.
* `camera_index`: `0` – OpenCV camera index.
* `force_cv2`: `false` – Force OpenCV capture pipeline.
* `inference_resolution`: `[224, 224]` – **Must match** the TFLite model’s expected input. Changing this without changing the model will break inference.

### `model`

* `model_path`: `./model/model_object.tflite` – Backbone used for features.
* `labels_path`: `./model/labels_object.txt` – Class names (one per line or in `idx name` format).
* `threshold`: `0.75` – Only relevant if you use the **raw TFLite classifier** head. Our k‑NN path doesn’t use this threshold.

### `tinylcm.feature_extractor`

* `type`: `TFLiteFeatureExtractor` – Use the on‑device TFLite wrapper.
* `model_path`: same as above.
* `feature_layer_index`: `-1` – Which output tensor to treat as features (−1 = last). If your model exports multiple tensors, pick the correct one.
* `normalize_features`: `false` – Set `true` if you want L2 normalization (useful with cosine distance).
* `lazy_loading`: `false` – If `true`, loads on first use (saves memory at boot).

**Changing**: Using a different `feature_layer_index` or model requires rebuilding the **feature\_processor** and **initial state** to keep dimensions consistent.

### `tinylcm.feature_transformation`

* `type`: `StandardScalerPCA` – Matches our export tool.
* `model_path`: `./model/feature_processor.npz` – Contains `μ`, `σ`, `W`.
* `format`: `npz` – Export format.
* `enabled`: `true` – Turn on/off post‑processing.
* `normalize_output`: `false` – L2‑normalize after PCA if desired.

**Changing**: If you swap this artifact or toggle `enabled`, you **must** recreate the initial k‑NN state to avoid **feature dimension mismatch**.

### `tinylcm.adaptive_classifier` (LightweightKNN)

* `k`: `5` – Neighbors for prediction; also affects distance aggregation. Larger `k` smooths but may mask fine changes.
* `max_samples`: `200` – Memory cap; oldest are evicted.
* `distance_metric`: `euclidean` – Or `cosine`.
* `weight_by_distance`: `true` – Use IDW for probabilities.
* `idw_power`: `2.0` – Higher emphasizes close neighbors.
* `initial_state_path`: `./initial_state/knn_initial_state_objects.json` – Must match the current feature pipeline.

**Changing**: If `k` changes, you should recompute **reference statistics** with the same `k` and aggregation mode.

### `tinylcm.drift_detectors[0]` (KNNDistanceMonitor)

* `delta`: `0.2` – Insensitivity band. If using adaptive thresholds, this is recomputed from `reference_std`.
* `lambda_threshold`: `5` – PH trigger threshold. Also recomputed if adaptive thresholds are enabled.
* `exit_threshold_factor`: `0.7` – Hysteresis to leave drift state.
* `high_confidence_threshold`: `0.75` – If classifier predicts a **stable** class above this, PH resets/ignores.
* `stable_known_classes`: `["stone", "leaf", "coin"]` – Classes that should **not** trigger drift when confidently seen; also used as **reference pool**.
* `reference_stats_path`: `./initial_state/knn_reference_stats.json` – Loads `reference_mean`/`std` for calibration.
* `warm_up_samples`: `0` – If no precomputed stats, collect this many samples to estimate `μ_ref`/`σ_ref` on‑device.
* `reference_update_interval`: `100` – How often to update `μ_ref` (EWMA).
* `reference_update_factor`: `0.02` – EWMA beta (`β`). Larger = adapt faster, but risk of drifting baseline.
* `pause_reference_update_during_drift`: `true` – Freeze the baseline when drift is active.
* `drift_cooldown_period`: `10` – Suppress back‑to‑back drift events (in frames).
* `use_adaptive_thresholds`: `true` – If `σ_ref` is available, compute `δ = α·σ_ref` and `λ = β·σ_ref`.
* `allow_runtime_threshold_adaptation`: `false` – If `true`, thresholds continue to evolve with updated `σ_ref`.
* `adaptive_delta_std_multiplier` (`α`): `0.5` (example) – Multiplier for `δ`.
* `adaptive_lambda_std_multiplier` (`β`): `5.0` – Multiplier for `λ`.
* `aggregation_mode`: `"q"` – Aggregate neighbor distances via quantile.
* `aggregation_q`: `0.9` – High‑quantile is robust to a few tight neighbors but still reacts to most distances increasing.
* `two_sided`: `true` – Detect both **up**/**down** shifts. Set **`false` for Lego scenario** to get a **one‑sided (up)** detector that fires only when distances get larger (novel object).

**Changing**:

* If you flip `two_sided → false`, only “distance increases” (unknown/novel) will trigger. Great for “Lego should cause drift, but closer‑than‑usual samples should not”.
* If you widen `aggregation_q` to 0.95+, the signal becomes more conservative; lower q (e.g. 0.5/median) is more sensitive to any near neighbor.
* Always keep **`k`, `aggregation_mode/q`, and reference stats** consistent.

### `tinylcm.data_logger`

* `enabled`: `true` – Writes per‑frame summaries and drift events.
* `log_dir`: `./logs`
* `log_images`: `false` – Can be turned on if you need visual debugging.
* `log_drift_events`: `true`
* `frame_log_interval`: `10` – Log every N frames to JSONL.
* `max_log_size_mb`: `100`

### `application`

* `inference_interval_ms`: `200` – Target spacing between frames. Lower → more CPU.
* `display_frames`: `false` – Draws UI; keep off on Pi Zero for performance.
* `save_debug_frames`: `false`
* `debug_output_dir`: `./debug`
* `log_each_prediction`: `true` – Useful during tuning; can be noisy.

---

## Runtime Flow (what happens per frame)

1. **Capture & resize** frame → `[224×224]` RGB.
2. **TFLiteFeatureExtractor** → feature vector.
3. **StandardScalerPCA** (optional) → transformed feature.
4. **LightweightKNN**:

   * Predict class & probability (IDW or counts).
   * Keep last `k` neighbor distances.
5. **KNNDistanceMonitor**:

   * Filter the reference pool to **stable classes** only.
   * Compute `k` distances, aggregate (e.g. `q=0.9`).
   * Update **Page‑Hinkley** (one‑sided or two‑sided).
   * If **drift**: log event (+ direction for two‑sided).
   * If **high‑confidence known class**: reset PH stats.
6. **Performance logging**: latency, confidence, CPU/mem, summaries every 100 frames.

---

## Tuning & Troubleshooting

**Model confusions**

* k‑NN probs depend on the stored samples. If the backbone features are weak, consider:

  * Better data augmentation; more class balance
  * Re‑export PCA with suitable target dim (e.g. 256)
  * Try cosine distance + L2 norm

**No drift fires when it should**

* Ensure `lego` is **excluded** from the reference pool and **two\_sided=false** for the Lego scenario.
* Recompute `knn_reference_stats.json` after changing `k`/aggregation.
* Use higher `β` (adaptive lambda multiplier) or lower `λ` for more sensitivity.

**Drift firing too often**

* Increase `λ` (or `β` if adaptive), or increase `δ`.
* Lower `aggregation_q` (e.g. 0.75) to be less influenced by a few large distances.
* Increase `high_confidence_threshold` so stable sightings reset PH more aggressively.

**Feature dimension mismatch**

* Happens if you switch models or PCA without recreating the initial state. Re‑run `make_feature_processor.py` and `create_initial_state.py`.

**Latency too high**

* Reduce camera resolution / FPS.
* Use smaller PCA dim.
* Lower `k`.

---

## Scenarios

### Scenario 1 – Lego causes drift

* Train WITHOUT Lego in the k‑NN state (exclude `lego` from initial samples).
* Use **one‑sided (up)** PH: set `two_sided: false`.
* Keep `stable_known_classes: ["stone", "leaf", "coin"]` and exclude `negative` from the reference statistics for a cleaner baseline.

### Scenario 2 – Everything known (sanity)

* Include all classes in the initial state.
* Use two‑sided to also catch data quality issues where distances become unusually **small** (e.g., near duplicates or camera changes making features collapse).

---

## Minimal Code References (short snippets)

**Aggregating neighbor distances**

```python
# d = sorted k-NN distances
if mode == 'q':
    agg = np.quantile(d, q)  # e.g., q=0.9
elif mode == 'median':
    agg = np.median(d)
else:
    agg = np.mean(d)
```

**Two‑sided Page‑Hinkley update**

```python
e = x_t - mu_ref
cum_up   = max(0.0, cum_up   + (e - delta))
cum_down = max(0.0, cum_down + (-e - delta))
if cum_up > lambda_ or cum_down > lambda_:
    drift = True
```

**EWMA reference update**

```python
mu_ref = (1 - beta) * mu_ref + beta * x_t
```

---

## File Outputs

* `initial_state/knn_initial_state_*.json` – k‑NN samples, labels, timestamps, params.
* `initial_state/knn_reference_stats.json` – `reference_mean`, `reference_std`, class stats, aggregation mode.
* `logs/drift_detection_*.log` – Rich per‑frame log with distances, PH stats.
* `logs/data_*.jsonl` – Compact frame summaries for downstream analysis.

---

## Glossary

* **Stable classes** – Known, trusted classes that act as the **reference** for distance statistics; high‑confidence sightings can reset PH.
* **One‑sided PH** – Detects only **increases** in aggregated distance (good for novel object discovery like Lego).
* **Two‑sided PH** – Detects both increases and decreases (useful to catch distribution collapses or unexpected easy matches).
* **EWMA** – Exponentially Weighted Moving Average; smooth reference mean update to track slow environmental changes.

---

## License

MIT
