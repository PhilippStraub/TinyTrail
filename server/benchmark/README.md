# Multi-Scenario Benchmark Pipeline

Automated benchmark runner for TinyTrail classifier scenarios. The pipeline can run one
or more scenarios sequentially, deploy each scenario to the Raspberry Pi via NATS, collect
the returned logs, and store results in scenario-specific folders.

The pipeline is configured in `server/benchmark/benchmark_config.json` and implemented in
`server/benchmark/benchmark_pipeline.py`.

## What Gets Swept

For every scenario listed in `experiment_names`, the pipeline runs the full cross-product:

```text
pca_configs x samples_per_class x seeds
```

With the current default config:

```text
2 PCA modes x 12 sample counts x 10 seeds = 240 experiments per scenario
```

If you list three scenarios, that becomes 720 experiments.

## Scenario List

Use `experiment_names` in `benchmark_config.json`:

```json
{
  "experiment_names": [
    "scenario_cifar10_knn",
    "scenario_cifar10_nearest_centroid",
    "scenario_cifar10_kmeans"
  ]
}
```

The pipeline also accepts the aliases `experiment_name`, `Experiment Name`,
`scenario_names`, `scenarios`, and `scenario`, but `experiment_names` is the recommended
field.

You can also use objects for per-scenario overrides:

```json
{
  "experiment_names": [
    "scenario_cifar10_nearest_centroid",
    {
      "name": "scenario_imagenette_nearest_centroid",
      "dataset_name": "imagenette",
      "test_images_per_class": 10,
      "pca_configs": [
        { "enabled": true, "components": 64 },
        { "enabled": true, "explained_variance": 0.95 },
        { "enabled": false }
      ]
    }
  ]
}
```

Object fields override the global benchmark config only for that scenario.

## Adding A New Scenario

Step-by-step checklist to add a scenario that reuses an already-trained model (no
retraining triggered by the pipeline itself):

1. **Create the scenario folder.** The pipeline requires:
   ```text
   server/Retrainer/on-device-drift-detection-main/examples/python_examples/scenario_<name>/
   ```
   If this folder does not exist, `build_scenario_context()` raises `FileNotFoundError`
   immediately. Copy an existing scenario of the same classifier type
   (`scenario_cifar10_knn`, `_nearest_centroid`, or `_kmeans`) as a starting
   point for `config.json`, `main.py`, and `create_initial_state.py`.

2. **Drop in your already-trained model.** Place these files under
   `scenario_<name>/model/`:
   ```text
   model_object.tflite
   labels_object.txt
   feature_processor.npz                     # optional, pre-fit scaler/PCA
   feature_processor_pca<N>.npz               # optional cache, fixed component count
   feature_processor_pca<pct>var.npz          # optional cache, e.g. pca95var for 95%
   ```
   If a `feature_processor_pca*` cache matching your `pca_configs` entry is already
   present, `ensure_feature_processor()` skips training a new one — see
   [Model Training And Feature Processor](#model-training-and-feature-processor).

3. **Point `create_initial_state.py` at your reference (train) images.** Edit the
   `CLASSES` dict / `INITIAL_IMAGE_DATA_DIR` at the top of the copied script, following
   the existing convention:
   ```text
   examples/dataset_<name>/train/<class>/*.jpg
   ```
   Also set `CLASSIFIER_TYPE` in that file to match `tinylcm.adaptive_classifier.type`
   in `config.json` (`"knn"`, `"nearest_centroid"`, or `"kmeans"`) — see
   [Supported Classifiers](#supported-classifiers). This script only samples images
   through the already-trained model/feature extractor; it does not train weights.

4. **Provide the held-out test split.** Either place images at the dataset-name
   convention:
   ```text
   examples/dataset_<name>/test/<class>/*.jpg
   ```
   or point `test_data_dir` / `test_data_dirs` in `benchmark_config.json` at wherever
   your test images live — see [Image Resolution And Datasets](#image-resolution-and-datasets).
   Test images are sampled automatically per experiment by `populate_test_images()`:
   for each class it deterministically shuffles the files in the test folder with
   `random.Random(seed + class_index)` and copies the first `test_images_per_class` of
   them into `raw_data/<name>/input_images/<class>/`, rebuilding that folder from
   scratch every run. Different seeds in the sweep therefore draw different (but
   reproducible) test images, expanding effective coverage across the sweep. If
   `test_data_dir` cannot be resolved, this step raises `FileNotFoundError` before any
   deployment happens.

5. **Register the scenario.** Add `"scenario_<name>"` (the exact folder name, not an
   alias) to `experiment_names` in `server/benchmark/benchmark_config.json` — see
   [Scenario List](#scenario-list). Double-check the name matches the folder exactly;
   a mismatch (e.g. `"scenario_cifar10_knn_benchmark"` vs. the actual
   `scenario_cifar10_knn` folder) fails with the same `FileNotFoundError` as
   step 1.

6. **Do not hand-create these — the pipeline (re)generates them every run:**
   ```text
   scenario_<name>/initial_state/*.json
   server/Deployment-Packager/nats-model-service/raw_data/scenario_<name>/
   ```

7. **Dry-run first:**
   ```powershell
   & $py benchmark_pipeline.py --dry-run
   ```
   to confirm the scenario resolves (paths, classifier type, dataset name, test data
   dir) before deploying anything to the Pi.

## Required Scenario Files

Each scenario name must correspond to a folder here:

```text
server/Retrainer/on-device-drift-detection-main/examples/python_examples/<scenario_name>/
```

Required files and folders:

```text
<scenario_name>/
  config.json
  main.py
  create_initial_state.py
  model/
    model_object.tflite
    labels_object.txt
    feature_processor.npz              # needed for PCA runs, or generated/cached
  initial_state/
    knn_initial_state_objects.json     # overwritten by create_initial_state.py
    knn_reference_stats.json           # overwritten by create_initial_state.py
```

The deployed payload is assembled automatically under:

```text
server/Deployment-Packager/nats-model-service/raw_data/<scenario_name>/
```

You normally do not edit this folder by hand. The pipeline creates or refreshes:

```text
raw_data/<scenario_name>/
  main.py
  create_initial_state.py
  config.json
  experiment_config.json
  utils/
  tinylcm_python/
  model/
  initial_state/
  input_images/
```

`utils/` and `tinylcm_python/` are copied from the Retrainer project automatically.
They are required on the Raspberry Pi because the deployed `main.py` imports them directly.

## Model Training And Feature Processor

If a scenario has its own feature processor builder, place it here:

```text
server/Retrainer/on-device-drift-detection-main/examples/model_training/<scenario_name>/make_feature_processor.py
```

If this file is missing, the pipeline searches for a compatible builder by dataset name,
for example a `*cifar10*/make_feature_processor.py`. It calls the builder with:

```text
--model-path <scenario>/model/model_object.tflite
--output-dir <scenario>/model
--pca-components <N or variance>
```

Use `components` for a fixed output dimension, for example `31`.
Use `explained_variance` for an adaptive PCA dimension, for example `0.95`
or `95` for 95% explained variance. Imagenette benchmarks should use this
mode when you want the PCA reduction to keep 95% variance.

Generated PCA processors are cached as:

```text
<scenario>/model/feature_processor_pca<N>.npz
<scenario>/model/feature_processor_pca95var.npz
```

## Scenario Configs

The active Pi config for a scenario comes from this priority order:

1. `server/benchmark/config_templates/<scenario_name>.json`
2. `examples/python_examples/<scenario_name>/config.json`
3. `raw_data/<scenario_name>/config.json`
4. legacy fallback: `server/benchmark/config_template.json`

Recommended setup: keep each scenario's real base config in its own
`examples/python_examples/<scenario_name>/config.json`. Use
`server/benchmark/config_templates/<scenario_name>.json` only when you need a benchmark-only
override.

The old `server/benchmark/config_template.json` is now only a fallback for older setups. It
is no longer the single source of truth.

During each experiment the pipeline patches only runtime-specific fields:

```text
tinylcm.feature_transformation.enabled
tinylcm.adaptive_classifier.max_samples
image_input.shuffle_seed
```

Everything else, including classifier type and image resolution, should be defined in the
scenario config.

## Pi Logging

Keep benchmark runs quiet on the Raspberry Pi terminal:

```json
"device": {
  "log_level": "INFO",
  "console_log_level": "ERROR"
}
```

`log_level` controls the log file under `./logs`. `console_log_level` controls what is
printed to the Pi terminal and should stay `ERROR` for performance-oriented benchmark
runs.

Long per-image lines with predictions, distances, drift state, and latency are disabled
by default. Enable them only for debugging:

```json
"data_logger": {
  "enabled": false,
  "log_images": false,
  "log_drift_events": false,
  "verbose_frame_logging": false
}
```

The benchmark artifacts (`performance_*.json`, `resource_usage_*.csv`, and
`merged_analysis_*.csv`) are still written.

## Supported Classifiers

Set the classifier in the scenario `config.json`:

```json
"adaptive_classifier": {
  "type": "LightweightKNN",
  "k": 3,
  "max_samples": 200,
  "distance_metric": "euclidean",
  "weight_by_distance": true,
  "idw_power": 3.0,
  "initial_state_path": "./initial_state/knn_initial_state_objects.json"
}
```

Supported values:

```text
LightweightKNN
NearestCentroidClassifier
KMeansPrototypeClassifier
```

For KMeans, add:

```json
"m_prototypes": 3
```

Important: `create_initial_state.py` must create a state with the same classifier type as
the scenario config. For example:

```python
CLASSIFIER_TYPE = "nearest_centroid"
```

or:

```python
CLASSIFIER_TYPE = "kmeans"
```

The filenames `knn_initial_state_objects.json` and `knn_reference_stats.json` are kept for
compatibility. They may contain Nearest Centroid or KMeans prototype states.

## Image Resolution And Datasets

Set inference resolution in each scenario config:

```json
"image_input": {
  "inference_resolution": [32, 32]
}
```

For ImageNet/Imagenette-style models this may be:

```json
"image_input": {
  "inference_resolution": [224, 224]
}
```

The pipeline uses the scenario's `supported_extensions` when copying test images, so
`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, or `.JPEG` can be supported by config.

Dataset name is inferred from the scenario name:

```text
scenario_cifar10_*      -> cifar10
scenario_imagenette_*   -> imagenette
scenario_imagenet_*     -> imagenet
```

Override it if needed:

```json
{
  "experiment_names": [
    {
      "name": "scenario_my_custom_name",
      "dataset_name": "imagenette"
    }
  ]
}
```

Default test split locations:

```text
cifar10     -> server/Retrainer/on-device-drift-detection-main/examples/dataset_cifar10/test
imagenette  -> server/Retrainer/on-device-drift-detection-main/examples/dataset_imagenette/test
imagenet    -> server/Retrainer/on-device-drift-detection-main/examples/dataset_imagenet/test
              fallback: dataset_imagenette/test
```

Override globally or per dataset:

```json
{
  "test_data_dir": "server/Retrainer/on-device-drift-detection-main/examples/dataset_cifar10/test",
  "test_data_dirs": {
    "cifar10": "server/Retrainer/on-device-drift-detection-main/examples/dataset_cifar10/test",
    "imagenette": "server/Retrainer/on-device-drift-detection-main/examples/dataset_imagenette/test",
    "scenario_custom": "path/to/custom/test"
  }
}
```

`test_data_dirs` keys can be dataset names or exact scenario names.

`benchmark_config.json` accepts `//` line comments and tolerates a trailing comma
before `]`/`}` (`load_benchmark_config()` strips both, respecting string literals like
`"nats://..."`) — comment out scenario names in `experiment_names` to run one dataset
at a time without deleting entries.

## Result Layout

Results are now separated by scenario:

```text
server/benchmark/results/
  benchmark_summary.json
  scenario_cifar10_knn/
    benchmark_summary.json
    baseline_000/
      baseline_summary.json
      baseline_000_run1/
        power_*.csv, resource_usage_*.csv, merged_analysis_*.csv, ...
      baseline_000_run2/
      baseline_000_run3/
    exp_001_pca31_spc2_s42/
      merged_analysis_*.csv
      resource_usage_*.csv
      performance_*.json
      drift_detection_*.csv
      power_exp_001_pca31_spc2_s42.csv
      experiment_metadata.json
    baseline_001/
      baseline_summary.json
      ...
  scenario_cifar10_nearest_centroid/
    benchmark_summary.json
    exp_001_pca31_spc2_s42/
      ...
  scenario_cifar10_kmeans/
    benchmark_summary.json
    exp_001_pca31_spc2_s42/
      ...
```

Root `benchmark_summary.json` summarizes all scenarios. Each scenario also gets its own
`benchmark_summary.json`.

A `baseline_NNN/` folder is deployed automatically before the first
`(pca, samples_per_class)` cell and after every subsequent one — 3 idle runs
(`application.baseline_mode=true`) whose mean power becomes the **local baseline** that
nets out idle draw for every experiment it brackets (`net_avg_power_W_local`, `EDP` in
the analysis). No manual idle-power measurement step is needed. See
[`analysis/README.md`](analysis/README.md) §2.1 for the `baseline_summary.json` field
reference and §5 for how the correction is computed.

## Result Attribution & Energy-Data Integrity (2026-07-20)

**Relevant for anyone reviewing the measurement infrastructure / power-energy setup:**
result messages from the Pi are now tagged and routed by their true run identity,
closing a race that could previously misattribute one experiment's result to
another. Read this before drawing conclusions from `power`/`energy` fields, or from
result folders produced before this date.

**Symptom (found in existing CIFAR-10/Imagenette result sets):** a handful of
`exp_*` folders contained the *previous* experiment's `merged_analysis` CSV
(byte-identical, same run_id) instead of their own — proven via causal-containment
checks in [`analysis/verify_power_coverage.py`](analysis/verify_power_coverage.py)
(the whole deploy→result cycle took ~4-5 s while the embedded resource log spanned
49-114 s, which is only possible if the ZIP belonged to an earlier, already-finished
run). Because the local power trace for the mis-slotted experiment only covers those
~4-5 s, `power.inference_window` (and therefore `energy_per_image_J`) was silently
missing for exactly those runs — see `EXCLUDED_EXPERIMENTS` in
[`analysis/analyze_benchmark.py`](analysis/analyze_benchmark.py).

**Root cause:** the server subscribes to `analysis.logs.<device_id>` with
`DeliverPolicy.NEW` *before* deploying the next experiment
(`benchmark_pipeline.py`, `_subscribe_deploy_collect()`). If the Pi's belated
transmission of the *previous* run's logs (triggered by `stop_current()` when the
next deployment ZIP lands, `client/deployment-receiver.py`) went out only after that
new consumer was already listening, the old ZIP looked, on arrival order alone,
like the new run's result — the server previously accepted the first ZIP containing
any `merged_analysis*` file as "the" result, with no check that it actually belonged
to the experiment being awaited.

**Fix (implemented, all three files touched together):**
- `client/transmitter.py` — every log transmission now reads the deployment's own
  `experiment_config.json` (already written per experiment by
  `benchmark_pipeline.py::prepare_deployment()`) and tags the NATS publish with
  `X-Experiment-Id` / `X-Scenario-Name` headers.
- `server/benchmark/benchmark_pipeline.py` — `handle_msg()` now routes every
  incoming ZIP to the experiment folder named by its own tag, and only treats it as
  *this* deploy's result if the tag matches. A message tagged for a different
  (already-completed) run is filed under that run's own folder instead of being
  discarded or misattributed — no message is ever dropped.
- `client/deployment-receiver.py` — `DeploymentManager` now sends each run's final
  logs exactly once (`transmit_current_once()`), so the natural-completion watcher
  and the next-deployment-arrived path can no longer both fire for the same run and
  race each other into the next run's listening window.

**Practical implication for measurement:** result sets collected *before* this fix
(all current `server/benchmark/results/**` data) may still contain the handful of
misattributed experiments listed in `EXCLUDED_EXPERIMENTS`
(`analysis/analyze_benchmark.py`) — those are already excluded from every aggregate.
All experiments are being fully re-run regardless, so no manual repair of old data
is planned. Result sets collected *after* this fix should not exhibit this failure
mode at all, since a stray/duplicate message can no longer land in the wrong
experiment's folder.

## Resume And Summary Rebuild

The benchmark is resume-safe by default:

```json
"skip_completed": true
```

A run is treated as completed only if its experiment folder contains both:

```text
experiment_metadata.json
merged_analysis*.csv
```

Folders that only contain a partial power log, for example after a timeout, are not
treated as completed. When the benchmark is started again, those combinations are
deployed again and the missing result files are written into the same `exp_*` folder.
Completed runs are skipped.

To rebuild the summaries from existing complete result folders without deploying
anything:

```powershell
& $py benchmark_pipeline.py --rebuild-summary
```

This rewrites:

```text
server/benchmark/results/benchmark_summary.json
server/benchmark/results/<scenario_name>/benchmark_summary.json
```

To intentionally run completed combinations again:

```powershell
& $py benchmark_pipeline.py --rerun-completed
```

Do not use `--clean-results` for a resume run. It deletes all `exp_*` folders,
including good completed runs.

Each `experiment_metadata.json` includes:

```json
{
  "scenario_name": "scenario_cifar10_nearest_centroid",
  "dataset_name": "cifar10",
  "classifier_type": "NearestCentroidClassifier",
  "config_source": ".../scenario_cifar10_nearest_centroid/config.json",
  "inference_resolution": [32, 32],
  "pca_enabled": true,
  "pca_components": null,
  "pca_target": 0.95,
  "pca_target_type": "explained_variance",
  "pca_label": "pca95var",
  "pca_explained_variance_%": 95.02,
  "samples_per_class": 2,
  "seed": 42,
  "power": {}
}
```

This is the field the analysis step can use later to group by dataset, scenario, and
classifier.

## Running

From `server/benchmark/`:

```powershell
$py = "C:\Users\jonas\DBE\KNN Sampling\tinymlops-framework\.venv311\Scripts\python.exe"
```

Dry run:

```powershell
& $py benchmark_pipeline.py --dry-run
```

Run all configured scenarios:

```powershell
& $py benchmark_pipeline.py
```

Large Imagenette deployments can exceed the NATS `max_payload` limit if sent as
one ZIP. The packager now automatically sends ZIPs larger than the single-message
limit in chunks, and `client/deployment-receiver.py` reassembles them on the Pi.
Make sure the Pi is running the updated receiver before launching Imagenette runs.

KNN deployments become especially large for Imagenette because the adaptive classifier
stores the selected support samples as feature vectors in `initial_state/`. The size
therefore grows with `classes x samples_per_class x feature_dimension`. Imagenette uses
a larger input resolution and a stronger feature extractor than CIFAR-10, so its
intermediate feature vectors are much wider. PCA reduces that feature dimension, which
is why `pca95var` deployments are smaller than no-PCA deployments, but high
`samples_per_class` values can still create multi-megabyte initial-state JSON files.
Nearest Centroid and KMeans usually stay smaller because they store aggregated
representatives such as centroids instead of every selected KNN support vector.

## Raspberry Pi Receiver Over SSH

The Pi does not need a display. Use SSH only to install, start, stop, or inspect the
receiver. For long benchmark runs, the receiver should run as a `systemd` service so it
does not depend on an open SSH terminal.

Copy the updated receiver from Windows to the Pi:

```powershell
scp client/deployment-receiver.py hans@<PI-IP>:/home/hans/client/deployment-receiver.py
```

Log into the Pi:

```powershell
ssh hans@<PI-IP>
```

Check that the receiver contains the large-message/chunk support:

```bash
grep -n "MAX_MSG_SIZE\|chunk" /home/hans/client/deployment-receiver.py
```

Create the service:

```bash
sudo nano /etc/systemd/system/deployment-receiver.service
```

Service file for the current Pi folder layout:

```ini
[Unit]
Description=TinyTrail Deployment Receiver
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/home/hans/client
ExecStart=/usr/bin/python3 /home/hans/client/deployment-receiver.py
Restart=always
RestartSec=5
User=hans

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable deployment-receiver
sudo systemctl restart deployment-receiver
sudo systemctl status deployment-receiver
```

Watch live logs:

```bash
journalctl -u deployment-receiver -f
```

After copying a newer `deployment-receiver.py`, restart the service:

```bash
sudo systemctl restart deployment-receiver
```

Also restart the service after **any** NATS broker restart (Docker container restart,
crash, host reboot) — the JetStream push-consumer subscription does not reliably survive
a broker restart even though the raw NATS connection itself reconnects. Do not rely on
automatic reconnect/resubscribe; a message can otherwise get stuck undelivered
(`num_pending` never reaching the consumer) until the receiver is restarted.

If you only need a short manual run, `tmux` is acceptable:

```bash
tmux new -s receiver
cd /home/hans/client
python3 deployment-receiver.py
```

Detach with `Ctrl+B`, then `D`; reattach with:

```bash
tmux attach -t receiver
```

Use a custom config or output folder:

```powershell
& $py benchmark_pipeline.py --config my_cfg.json --results-dir results_run2
```

Clean previous `exp_*` folders before a fresh run:

```powershell
& $py benchmark_pipeline.py --clean-results
```

The idle-power (baseline) correction no longer needs a manual measurement step — the
pipeline automatically deploys 3 idle runs before the first and after every
`(pca, samples_per_class)` cell and derives the net power from those bracketing blocks
(see `analysis/README.md` §5).

## Current Default Config

Current `benchmark_config.json` runs:

```json
"experiment_names": [
  "scenario_cifar10_knn",
  "scenario_cifar10_kmeans",
  "scenario_cifar10_nearest_centroid",
  "scenario_imagenette_knn",
  "scenario_imagenette_kmeans",
  "scenario_imagenette_nearest_centroid",
  "scenario_rover-lab_knn",
  "scenario_rover-lab_kmeans",
  "scenario_rover-lab_nearest_centroid"
]
```

To benchmark KNN, Nearest Centroid, and KMeans sequentially:

```json
"experiment_names": [
  "scenario_cifar10_knn",
  "scenario_cifar10_nearest_centroid",
  "scenario_cifar10_kmeans"
]
```

## Files In `server/benchmark`

```text
benchmark_pipeline.py     multi-scenario runner
benchmark_config.json     scenario list, sweep dimensions, power config
config_template.json      legacy fallback config
config_templates/         optional per-scenario benchmark-only configs
dsp3005_reader.py         Voltcraft DSP-3005 reader
results/                  benchmark outputs
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Scenario not found | `examples/python_examples/<scenario_name>/` must exist |
| Packager folder missing | OK, the pipeline creates `raw_data/<scenario_name>/` |
| Wrong classifier runs | Check `tinylcm.adaptive_classifier.type` and `CLASSIFIER_TYPE` in `create_initial_state.py` |
| Wrong input resolution | Set `image_input.inference_resolution` in the scenario config or scenario template |
| `ModuleNotFoundError: No module named 'utils'` on Pi | Rerun the benchmark pipeline so `utils/` and `tinylcm_python/` are copied into `raw_data/<scenario_name>/` before packaging |
| Test split missing | Set `test_data_dir` or `test_data_dirs` in `benchmark_config.json` |
| KMeans PCA builder missing | Add a scenario-specific `make_feature_processor.py` or rely on the dataset fallback |
| `nats.errors.MaxPayloadError` | Use the updated packager and updated Pi `client/deployment-receiver.py`; large deployments are sent chunked |
| Timeout after deployment | Check that `deployment-receiver.service` is running on the Pi and watch `journalctl -u deployment-receiver -f` |
| Result folders mixed | Results are separated under `results/<scenario_name>/exp_*`; use `--clean-results` for a fresh sweep |
| Deployment message stuck (`num_pending` never delivered) after a NATS broker restart | The JetStream push-consumer subscription does not reliably survive a broker restart even though the raw NATS connection reconnects (`reconnected_cb` fires, but delivery can stay stuck). Restart `deployment-receiver.service` on the Pi to force a fresh subscription — do not rely on automatic reconnect/resubscribe. Restart the Pi service after every NATS restart (Docker restart, crash, host reboot), not just after copying new code. |
