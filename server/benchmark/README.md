# KNN Sweet-Spot Benchmark Pipeline

Automated measurement environment for the DBE SS26 project *"Analyse und Identifikation
des Sweet Spots für die Sample-Größe des KNNs im TinyMLOps Framework."*

It answers three research questions empirically:

1. **Sweet spot** — how does KNN accuracy scale with the number of reference samples?
2. **PCA usefulness** — does PCA help (accuracy / memory / CPU / energy) vs. raw features?
3. **Cost** — how do memory, CPU and energy grow with sample size and PCA?

For every configuration it trains the model artifacts on the server, deploys them to a
Raspberry Pi Zero 2W over NATS, runs inference on the Pi, measures power externally with a
Voltcraft DSP-3005, pulls the results back, and stores everything for analysis.

---

## 1. How it works (one experiment)

```
                         ┌──────────────────────── this PC (server) ───────────────────────┐
                         │                                                                  │
  benchmark_config.json ─┤  benchmark_pipeline.py                                           │
                         │     1. ensure feature_processor_pcaN.npz      (cached)           │
                         │     2. create_initial_state.py  --seed S      → KNN state JSON    │
                         │     3. patch config.json + copy artifacts → raw_data/            │
                         │     4. SUBSCRIBE  analysis.logs.<device>      (DeliverPolicy.NEW) │
                         │     5. START power logger (Voltcraft DSP-3005 on COMx)           │
                         │     6. DEPLOY: packager zips raw_data/ → publishes to NATS MODELS │
                         │                                                                  │
                         │   Voltcraft ── powers ──► Raspberry Pi ◄── USB-serial ── reads ──┘
                         │                                │
                         └────────────────────────────────┼─────────────────────────────────
                                                          │ NATS JetStream
                         ┌──────────────────────── Raspberry Pi Zero 2W ───────────────────┐
                         │  deployment-receiver.py                                          │
                         │     • receives ZIP, extracts, runs main.py                       │
                         │     • main.py: TFLite → (PCA) → LightweightKNN → drift monitor   │
                         │       writes merged_analysis_*.csv + resource_usage_*.csv        │
                         │     • on completion → transmitter.py zips logs → analysis.logs.* │
                         └──────────────────────────────────────────────────────────────────┘
                                                          │
                         ┌────────────────────────────────┼─────────────────────────────────
                         │  benchmark_pipeline.py                                           │
                         │     7. RECEIVE log ZIP, extract merged_analysis + resource CSV   │
                         │     8. STOP power logger                                         │
                         │     9. compute energy over the inference window                  │
                         │    10. write results/<experiment_id>/                            │
                         └──────────────────────────────────────────────────────────────────┘
```

The pipeline **subscribes before it deploys** (with `DeliverPolicy.NEW`), so the result it
receives is guaranteed to belong to the experiment it just launched. The old
`Analytical-Data-Processor` service is **not** required.

---

## 2. The sweep

The experiment list is the full cross-product defined in `benchmark_config.json`:

```
pca_configs  ×  samples_per_class  ×  seeds
```

- **pca_configs** — e.g. `{components:80}`, `{components:40}`, `{enabled:false}` (no-PCA)
- **samples_per_class** — reference samples per class loaded into the KNN, e.g. `2 … 100`
- **seeds** — repeats each (pca, spc) pair N times with fixed seeds so results are not a
  lucky draw of reference samples. The **same seed** controls both the reference-sample
  selection (`create_initial_state.py`) and the test-image shuffle (`config.json`).

Total experiments = `len(pca_configs) × len(samples_per_class) × len(seeds)`.
With the default config that is `3 × 8 × 5 = 120`.

> **Reference-state creation is fast and in-process.** For each experiment the pipeline calls
> `build_initial_state()` **in-process** instead of spawning a subprocess: the scenario module —
> and TensorFlow — is imported **once** for the whole sweep (so the pipeline process itself holds
> TF, a few hundred MB) and one TFLite interpreter is reused. It also sub-samples each class to
> `samples_per_class` **before** extracting features (the `train/` folders hold 5000 images/class,
> and only the kept samples are ever run through the model). Net effect: init costs seconds per
> experiment, not minutes, while the selection stays fully seed-reproducible.

Each experiment gets an id like `exp_001_pca80_spc2_s42`
(`<index>_<pca|nopca>_spc<N>_s<seed>`).

---

## 3. Prerequisites

| Requirement | Notes |
|---|---|
| Python venv | `.venv311` at the repo root, with `pyserial`, `nats-py`, `numpy`, `tensorflow` |
| Backbone model | `model_object.tflite` + `labels_object.txt` already trained (in the scenario `model/` dir) |
| CIFAR-10 dataset | `examples/dataset_cifar10/train` for reference-sample extraction |
| Test images | `raw_data/.../input_images/<class>/*.png` — the **test split** (novelty), per the paper |
| NATS server | running and reachable from both this PC and the Pi |
| Pi | `deployment-receiver.py` running (latest version with `watch_process_completion`) |
| Voltcraft DSP-3005 | (optional) on a COM port; the Pi is powered **through** it |

> **Clock sync (for energy):** the energy metric slices the power trace to the Pi's actual
> inference window using the Pi's `timestamp_unix`. Enable NTP on the Pi so its clock matches
> this PC: `sudo timedatectl set-ntp true`.

---

## 4. Configuration — `benchmark_config.json`

```jsonc
{
  "device_id": "my_device_01",          // NATS subject suffix; must match the Pi's device_id
  "nats_url": "nats://localhost:4222",  // broker URL (this PC's view)
  "result_timeout_seconds": 1800,       // max wait per experiment before marking TIMEOUT

  "pca_configs": [
    { "enabled": true,  "components": 80 },
    { "enabled": true,  "components": 40 },
    { "enabled": false }
  ],

  "samples_per_class": [2, 5, 10, 20, 30, 50, 75, 100],

  "seeds": [42, 123, 456, 789, 1337],

  "power": {
    "enabled": true,            // set false to skip energy logging
    "port": "COM3",             // Voltcraft DSP-3005 serial port
    "rate": 5.0,                // power samples per second
    "baseline_power_w": null    // idle Pi power (W) — fill in via --measure-baseline
  }
}
```

`config_template.json` is the **frozen base** Pi config. The pipeline loads it, patches only
the per-experiment fields (PCA on/off, `max_samples`, `shuffle_seed`) and writes the result
to `raw_data/.../config.json` for deployment. The template itself is never modified.

---

## 5. Running it

All commands run from `server/benchmark/` using the repo venv python.

```powershell
# (Windows / PowerShell — adjust the python path if needed)
$py = "C:\Users\jonas\DBE\KNN Sampling\tinymlops-framework\.venv311\Scripts\python.exe"
```

### Step 0 — dry run (no execution, just the plan)
```powershell
& $py benchmark_pipeline.py --dry-run
```

### Step 1 — measure the idle power baseline (once)
Boot the Pi, make sure `deployment-receiver.py` is running, but **no inference is running**.
```powershell
& $py benchmark_pipeline.py --measure-baseline 60
```
Copy the printed `P_idle` value into `power.baseline_power_w` in the config. This lets the
pipeline report the **net** (marginal) energy of inference, with the Pi's base draw removed.

### Step 2 — run the full sweep
```powershell
& $py benchmark_pipeline.py
```
Optional:
```powershell
& $py benchmark_pipeline.py --config my_cfg.json --results-dir results_run2
& $py benchmark_pipeline.py --clean-results       # fresh sweep: delete old exp_* first
```

You can interrupt (`Ctrl+C`) and re-run; completed experiment folders are simply overwritten
on the next pass if you run the same ids again.

**`--clean-results`** deletes every `exp_*` folder before the sweep starts (it leaves
`results/analysis/` and the baseline untouched). Use it for a clean run after changing the
config — when sample sizes or pipelines shift, old folders with now-unused index/spc
combinations would otherwise linger and get mixed into the analysis (which crawls every
`exp_*`). It is **opt-in on purpose**: without it the pipeline preserves the `Ctrl+C`-then-resume
behavior above, so a normal re-run never silently wipes a half-finished sweep.

---

## 6. Output layout

```
results/
  baseline_power.csv                     # from --measure-baseline
  benchmark_summary.json                 # per-experiment OK/TIMEOUT roll-up
  exp_001_pca80_spc2_s42/
    merged_analysis_*.csv                # one row per inference (accuracy, latency, drift, …)
    resource_usage_*.csv                 # Pi CPU% / RAM at 1 Hz, with absolute timestamps
    performance_*.json                   # summary written by the Pi
    power_exp_001_pca80_spc2_s42.csv     # RAW Voltcraft readings (timestamp,V,I,P,…)
    experiment_metadata.json             # config + PCA variance + derived power/energy
  exp_002_pca80_spc2_s123/
    ...
```

### `experiment_metadata.json`
```jsonc
{
  "pca_enabled": true,
  "pca_components": 80,
  "samples_per_class": 2,
  "seed": 42,
  "experiment_id": "exp_001_pca80_spc2_s42",
  "started_at": "...",
  "result_received_at": "...",
  "pca_explained_variance_%": 94.3,      // variance retained by PCA (from the .npz)
  "power": {
    "full_window":      { "energy_J": ..., "avg_power_W": ..., "duration_s": ... },
    "inference_window": {                 // sliced to the Pi's actual inference period
      "energy_J": ..., "avg_power_W": ..., "energy_per_image_J": ...,
      "net_avg_power_W": ...,             // minus idle baseline = marginal inference cost
      "net_energy_per_image_J": ...
    }
  }
}
```

---

## 7. The metrics, and which column feeds which plot

| Metric | Source | Column(s) |
|---|---|---|
| Accuracy | `merged_analysis_*.csv` | `cumulative_accuracy`, `final_accuracy`, `correct` |
| Latency | `merged_analysis_*.csv` | `latency_ms` |
| Confidence / drift | `merged_analysis_*.csv` | `confidence`, `drift_detected`, `ground_truth`, `prediction` |
| CPU % | `resource_usage_*.csv` | `cpu_percent_total` (native BLAS + TFLite threads pinned to 1 — see §8) |
| Memory | `resource_usage_*.csv` | `process_rss_mb`, `memory_percent_total` |
| Energy / Power | `power_*.csv` → `experiment_metadata.json` | `P_out` → `energy_per_image_J`, `net_*` |

---

## 8. Scientific notes (measurement validity)

- **Energy is measured out-of-band.** The Voltcraft is read by *this PC*, not the Pi, so the
  power figures carry **zero measurement load** on the device — they represent what a real
  deployment (e.g. a rover) would draw.
- **On-Pi instrumentation is constant.** The only on-device logging is `psutil` at 1 Hz (for
  CPU/RAM). It costs the same regardless of sample size or PCA, so it shifts every curve by a
  fixed offset and does **not** distort the sweet-spot trend. Quantify it once with a
  calibration run (psutil on vs off) and document the offset.
- **Net energy isolates inference.** Subtracting `baseline_power_w` removes the Pi's idle draw
  and the fixed instrumentation cost, leaving the marginal cost attributable to sample size
  and PCA — exactly the paper's Energy curve.
- **Native threads are pinned to 1 (BLAS + TFLite).** Without pinning, the pca80 pipeline pegged
  ~3 cores (`process_cpu` ~315 %, system CPU ~79 %, **flat across all sample counts, even spc=2**)
  while pca40/nopca stayed single-threaded — inflating its CPU and energy ~16× with no latency
  benefit at this ~5 fps duty cycle, and making the pipelines incomparable. Because >100 % CPU
  needs *native* threads and the TFLite backbone + OpenCV preprocessing are identical across all
  pipelines, the culprit is **numpy's BLAS (OpenBLAS)**: its worker pool spins up and busy-waits,
  and only the pca80 config trips it. The fix is to pin the **BLAS/OpenMP** threads — the deployed
  `main.py` sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/… `=1` **before numpy is imported**
  (`os.environ.setdefault`). The TFLite interpreter is *also* pinned (`num_threads=1` in
  `tinylcm_python/feature_extractor.py`) as a secondary control. Both match the stationary-rover
  use case where inference is the only load; raise them only if the production rover would run
  multi-threaded. **Verify on the Pi after re-running:** a pinned pca80 run should drop from
  ~315 % to ~20 % `process_cpu`, matching pca40/nopca.
- **Reproducibility.** Every experiment is fully determined by its seed (reference-sample
  selection **and** test shuffle). Repeating with the same `seeds` list reproduces results;
  averaging across seeds gives the mean ± spread for each (pca, spc) point.

---

## 9. Standalone power logging (manual)

The reader also works on its own, independent of the pipeline:

```powershell
& $py dsp3005_reader.py COM3                 # log at 5 Hz to dsp3005/dsp3005_log_<ts>.csv
& $py dsp3005_reader.py COM3 --rate 3
& $py dsp3005_reader.py COM3 --csv out.csv
```

---

## 10. Files in this folder

| File | Purpose |
|---|---|
| `benchmark_pipeline.py` | Orchestrator — runs the whole sweep |
| `benchmark_config.json` | Sweep definition (pca × spc × seeds) + power settings |
| `config_template.json` | Frozen base Pi config; patched per experiment |
| `dsp3005_reader.py` | Voltcraft DSP-3005 serial reader + `PowerLogger` class |
| `results/` | Per-experiment output |

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `filtered consumer not unique on workqueue stream` | Old `analytical-receiver` consumer; pipeline recreates the ANALYSIS stream with `limits` retention automatically on first run. |
| `consumer must be deliver all on workqueue stream` | Same root cause — stream was `workqueue`; first run converts it to `limits`. |
| Pipeline waits forever | Pi's `deployment-receiver.py` not running, or it's an old version without `watch_process_completion`; results then only transmit on the 300 s timer. |
| `ModuleNotFoundError: utils` on the Pi | `utils/` + `tinylcm_python/` must be bundled into `raw_data/`; already copied — re-deploy. |
| `[POWER] Could not start power logger` | Wrong `port`, device off, or the *"Power Supply PC Monitor"* app holding the COM port. Close it. A power error never aborts the sweep. |
| Empty `inference_window` energy | Pi/PC clocks not synced — enable NTP on the Pi. `full_window` is still recorded. |
| `has Serial: False` | The wrong `serial` package is installed; use `pyserial` (`pip uninstall serial && pip install pyserial`). |
