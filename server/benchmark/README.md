# Multi-Scenario Benchmark Pipeline

Automated benchmark runner for TinyMLOps classifier scenarios. The pipeline can run one
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
    "scenario_cifar10_knn_benchmark",
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

## Result Layout

Results are now separated by scenario:

```text
server/benchmark/results/
  benchmark_summary.json
  scenario_cifar10_knn_benchmark/
    benchmark_summary.json
    exp_001_pca31_spc2_s42/
      merged_analysis_*.csv
      resource_usage_*.csv
      performance_*.json
      drift_detection_*.csv
      power_exp_001_pca31_spc2_s42.csv
      experiment_metadata.json
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
Description=TinySphere Deployment Receiver
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

Measure idle Pi power:

```powershell
& $py benchmark_pipeline.py --measure-baseline 60
```

Copy the printed idle power into:

```json
"power": {
  "baseline_power_w": 0.6140
}
```

## Current Default Config

Current `benchmark_config.json` runs:

```json
"experiment_names": [
  "scenario_imagenette_knn",
  "scenario_cifar10_knn_benchmark",
  "scenario_imagenette_nearest_centroid",
  "scenario_cifar10_nearest_centroid",
  "scenario_imagenette_kmeans",
  "scenario_cifar10_kmeans"
]
```

To benchmark KNN, Nearest Centroid, and KMeans sequentially:

```json
"experiment_names": [
  "scenario_cifar10_knn_benchmark",
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
