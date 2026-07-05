# Benchmark Analysis — measurements & results

What the multi-scenario sweet-spot analysis measures, how each number is collected, and how to
read the dataset notebooks. This is the companion to
[`benchmark_analysis_cifar10.ipynb`](benchmark_analysis_cifar10.ipynb),
[`benchmark_analysis_imagenet.ipynb`](benchmark_analysis_imagenet.ipynb), and
[`analyze_benchmark.py`](analyze_benchmark.py) (the headless engine and shared extraction logic).

> **Two layers, one extraction logic.** The notebook imports `analyze_benchmark.py`, so the
> interactive figures and the `python analyze_benchmark.py` report are computed identically.
> Run the engine from a terminal to regenerate `experiment_summary.csv`, the figures, and
> `analysis_report.md` without opening Jupyter.

## Dataset-specific notebooks

The multi-scenario benchmark is split into two interactive notebooks so dataset families do not
overwrite each other's exports:

- `benchmark_analysis_cifar10.ipynb` analyzes all `scenario_cifar10_*` runs together.
- `benchmark_analysis_imagenet.ipynb` analyzes the ImageNet-style/Imagenette runs
  (`scenario_imagenette_*`) together.

Both notebooks group by scenario, classifier, PCA mode, and `samples_per_class`, integrate the
power boxplot from `plot_power_boxplot.py`, and write each run to a timestamped folder:

```text
exports/
  cifar10/analysis_<timestamp>/
  imagenet/analysis_<timestamp>/
```

This keeps CSVs, figures, and generated findings distinguishable across datasets and reruns.
The exported findings include a project-requirements checklist aligned with the task PDF
`01 knn-sweet-spot-1.pdf`: hardware, task, datasets, PCA/no-PCA model variants,
reference-sample sweep, metric collection, training/benchmark process, PCA variance,
statistics, and the expected PCA/no-PCA result graphs.

---

## 1. Experimental setup

| Item | Value |
|---|---|
| Hardware (DUT) | Raspberry Pi Zero 2W |
| Power instrument | Voltcraft DSP-3005 bench supply, logged over serial (COM3) at ~5 Hz |
| Supply rail | ~5.01 V, stable (min/mean/max 5.009 / 5.010 / 5.010 V) |
| Dataset | CIFAR-10 and Imagenette, **test split** = novelty samples |
| Feature extractor | TFLite embedding; 160-D for CIFAR-10, 1280-D for Imagenette |
| Dimensionality reduction | PCA by target variance (`pca95var`) or `nopca` (raw feature vector) |
| Classifier | LightweightKNN, NearestCentroidClassifier, KMeansPrototypeClassifier |
| Sweep | scenarios × PCA modes × `samples_per_class` × seeds, one folder per run under `../results/<scenario_name>/exp_*/` |
| Replicates | the **seed** controls *both* reference-sample selection and test-image order |

The pipeline is trained on the **full** training set; only the **initialization sample size**
(`samples_per_class`) is varied in the benchmark.

---

## 2. Raw data per experiment (`../results/<scenario_name>/exp_*/`)

| File | Contents used |
|---|---|
| `experiment_metadata.json` | config (pca, samples_per_class, seed), PCA explained variance, and the pre-computed **`power`** block (energy/charge metrics) |
| `merged_analysis_*.csv` | one row per inference: prediction, ground_truth, confidence, latency, cumulative/final accuracy, CPU%, process RSS, `timestamp_unix`, drift flag |
| `power_*.csv` | raw Voltcraft trace: `timestamp, V_set, I_set, V_out, I_out, P_out, output` (~5 Hz) |
| `resource_usage_*.csv` | Pi-side resource log; its `timestamp_unix` `[min,max]` defines the **inference window** |

---

## 3. The tidy table (`summary`) — one row per experiment

Built by `build_summary()`. Columns and how each is derived:

| Column | Unit | Meaning / how collected |
|---|---|---|
| `experiment_id`, `seed` | — | run identity |
| `scenario_name`, `classifier_label`, `dataset_name` | — | scenario identity and classifier family |
| `pca_enabled`, `pca_components`, `pca_label` | — | pipeline (`nopca`, `pca95var`, fixed component counts, …) |
| `samples_per_class` | count | reference samples or prototypes initialized **per class** |
| `pca_explained_variance_pct` | % | variance retained by PCA (from the `.npz`) |
| `n_images` | count | test images inferred in the run (≈ 200) |
| `accuracy` | fraction | final cumulative accuracy over the test stream |
| `latency_ms_mean`, `latency_ms_p95` | ms | per-inference wall-clock latency on the Pi |
| `cpu_percent_mean` | % | system-wide CPU, **duty-cycled** at the rover cadence (first sample dropped) |
| `memory_rss_mb_mean`, `_max` | MB | process RSS — **dominated by the ~60 MB Python+TFLite runtime** |
| `knn_store_kb` | KB | deterministic analytical reference-store estimate = `spc × classes × feature_dim × 4 B` for sample-storing classifiers |
| `drift_events` | count | drift detections (drift detection is otherwise out of scope) |
| `avg_power_W` | W | mean Pi power over the inference window (incl. idle) |
| `net_avg_power_W` | W | `avg_power_W − idle_baseline` → marginal draw of inference |
| `energy_per_image_J` | J | gross energy per inference (incl. idle) |
| `net_energy_per_image_J` | J | **marginal** energy per inference (idle removed) — the headline energy metric |
| `inference_duration_s` | s | length of the measured inference window |

The notebooks additionally derive `charge_run_mAh` and `net_charge_run_mAh` (charge per run).

---

## 4. Aggregation & statistics

`aggregate()` groups by scenario/classifier, PCA label, and `samples_per_class`, then averages
over the seeds:

- **mean** across seeds (the plotted line),
- **std** (ddof=1),
- **95% CI of the mean** — `t.ppf(0.975, n−1) × SEM` (Normal `1.96×SEM` fallback if SciPy absent); drawn as the shaded band.

**Sweet spot** (`find_sweet_spot`): the smallest `samples_per_class` whose mean accuracy reaches
**99 %** of that pipeline's maximum mean accuracy (the knee).

**PCA effect** (`pca_effect`): **paired** PCA vs no-PCA at matched `(samples_per_class, seed)`,
tested with the **Wilcoxon signed-rank** test (≥ 5 pairs).

---

## 5. Energy methodology — why Joules, why *net*, why *per image*

The DSP-3005 reports instantaneous **power** `P_out` (Watts) for the whole Pi. Energy is the
integral of power over time:

```
E = ∫ P dt   ≈   Σ (P_i + P_{i+1})/2 · (t_{i+1} − t_i)     [trapezoidal]
```

- **Joule (J)** is the SI unit of **energy**: `1 J = 1 watt-second (W·s)`. Power (W) × time (s)
  = energy (J). The `_J` suffix on a column and the `(J)` on an axis label simply state that
  unit — e.g. `net_energy_per_image_J` is *Joules per image*.
- **Inference window.** The integral is sliced to the Pi's actual inference period (the
  `[min,max]` of `resource_usage_*.csv:timestamp_unix`), so boot/deploy time is excluded. If
  that window is unavailable (clock-sync gap) the code falls back to the full trace and records
  a `note`.
- **per image** = `energy_J / n_images`. Normalizes runs of different duration / image count
  into a fair **cost of one inference**.
- **net** = baseline-subtracted. The Pi draws power just being **on** (idle ≈ 0.61 W of the
  ≈ 0.79 W total). That idle offset is identical for every config and would swamp the small
  inference cost, so the idle baseline (measured once via
  `benchmark_pipeline.py --measure-baseline`) is subtracted:
  `net_avg_power_W = avg_power_W − baseline`, then `net_energy_per_image_J = net_avg_power_W ×
  duration / n_images`. This isolates the **marginal** energy that actually depends on sample
  size / PCA — exactly what the sweet-spot question needs.

The notebooks plot the **net** value. The absolute magnitude differs strongly by dataset and
feature extractor, so the exported findings report CIFAR-10 and Imagenette separately.

---

## 6. Charge used per run — mAh (§9)

Charge (mAh) is the intuitive battery unit. Because the rail is a near-constant 5 V, charge
converts directly from energy:

```
Q [mAh] = E [J] / (V × 3.6)        1 mAh = 3.6 C = V × 3.6 J = 18 J  (at 5 V)
```

The notebooks plot the **charge consumed per run** (one run = the ~200-image inference stream)
vs samples-per-class for each pipeline, both **gross** (idle included) and **net** (marginal,
idle removed). mAh figures are at the **5 V** rail. We make **no battery-capacity assumption** —
divide any pack's mAh rating by these per-run figures to get runtime.

The per-run charge curve also surfaces whether a configuration's energy cost scales with the
reference sample count or is dominated by fixed feature extraction / PCA overhead.

---

## 7. Exports

The dataset notebooks write tables, figures, and generated findings to timestamped export
folders:

```
exports/
  cifar10/analysis_<timestamp>/
    experiment_summary_cifar10.csv
    aggregated_by_scenario_cifar10.csv
    sweet_spots_95pct_cifar10.csv
    sweet_spots_99pct_cifar10.csv
    pca_effect_cifar10.csv
    project_requirements_cifar10.csv
    findings_cifar10.md
    figures/*.png
  imagenet/analysis_<timestamp>/
    experiment_summary_imagenet.csv
    aggregated_by_scenario_imagenet.csv
    sweet_spots_95pct_imagenet.csv
    sweet_spots_99pct_imagenet.csv
    pca_effect_imagenet.csv
    project_requirements_imagenet.csv
    findings_imagenet.md
    figures/*.png
```

The final export sets used for the write-up are:

- `exports/cifar10/analysis_20260701_230135/`
- `exports/imagenet/analysis_20260701_230136/`

---

## 8. Notebook section map

| § | What |
|---|---|
| 1–2 | load tidy data, aggregate across seeds |
| 3 | project curves: accuracy / net-energy / memory / CPU vs samples (PCA vs no-PCA, CI bands) |
| 4 | sweet spot + paired PCA effect |
| 5 | seed variability |
| 6 | per-experiment deep dive (latency / accuracy / confidence timeline + confusion matrix) |
| 7 | raw power trace for one experiment |
| **8** | **export tables, figures, project checklist, and generated findings** |

## 9. Reproduce

```bash
# headless: regenerate generic CSVs, figures, analysis_report.md
python analyze_benchmark.py
# or interactively, run the dataset notebooks top-to-bottom (re-crawls results/ live)
```
