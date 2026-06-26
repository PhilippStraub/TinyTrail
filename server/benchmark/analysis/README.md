# Benchmark Analysis — measurements & results

What the KNN sweet-spot analysis measures, how each number is collected, and how to read the
notebook. This is the companion to [`benchmark_analysis.ipynb`](benchmark_analysis.ipynb)
(interactive surface) and [`analyze_benchmark.py`](analyze_benchmark.py) (the headless engine —
single source of truth for metric extraction).

> **Two layers, one extraction logic.** The notebook imports `analyze_benchmark.py`, so the
> interactive figures and the `python analyze_benchmark.py` report are computed identically.
> Run the engine from a terminal to regenerate `experiment_summary.csv`, the figures, and
> `analysis_report.md` without opening Jupyter.

---

## 1. Experimental setup

| Item | Value |
|---|---|
| Hardware (DUT) | Raspberry Pi Zero 2W |
| Power instrument | Voltcraft DSP-3005 bench supply, logged over serial (COM3) at ~5 Hz |
| Supply rail | ~5.01 V, stable (min/mean/max 5.009 / 5.010 / 5.010 V) |
| Dataset | CIFAR-10 (10 classes), **test split** = novelty samples |
| Feature extractor | TFLite embedding → **160-D** feature vector |
| Dimensionality reduction | PCA (`pcaN`) reduces 160-D → N components, or `nopca` (raw 160-D) |
| Classifier | LightweightKNN (SVM / K-means are planned, see §7) |
| Sweep | pipelines × `samples_per_class` × seeds, one folder per run under `../results/exp_*/` |
| Replicates | the **seed** controls *both* reference-sample selection and test-image order |

The pipeline is trained on the **full** training set; only the **initialization sample size**
(`samples_per_class`) is varied in the benchmark.

---

## 2. Raw data per experiment (`../results/exp_*/`)

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
| `pca_enabled`, `pca_components`, `pca_label` | — | pipeline (`nopca`, `pca31`, …) |
| `samples_per_class` | count | reference samples stored **per class** in the KNN |
| `pca_explained_variance_pct` | % | variance of the 160-D features retained by PCA (from the `.npz`) |
| `n_images` | count | test images inferred in the run (≈ 200) |
| `accuracy` | fraction | final cumulative accuracy over the test stream |
| `latency_ms_mean`, `latency_ms_p95` | ms | per-inference wall-clock latency on the Pi |
| `cpu_percent_mean` | % | system-wide CPU, **duty-cycled** at the rover cadence (first sample dropped) |
| `memory_rss_mb_mean`, `_max` | MB | process RSS — **dominated by the ~60 MB Python+TFLite runtime** |
| `knn_store_kb` | KB | **deterministic** store size = `spc × classes × feature_dim × 4 B` (the part that actually scales with samples) |
| `drift_events` | count | drift detections (drift detection is otherwise out of scope) |
| `avg_power_W` | W | mean Pi power over the inference window (incl. idle) |
| `net_avg_power_W` | W | `avg_power_W − idle_baseline` → marginal draw of inference |
| `energy_per_image_J` | J | gross energy per inference (incl. idle) |
| `net_energy_per_image_J` | J | **marginal** energy per inference (idle removed) — the headline energy metric |
| `inference_duration_s` | s | length of the measured inference window |

The notebook (§9) additionally derives `charge_run_mAh` and `net_charge_run_mAh` (charge per run).

---

## 4. Aggregation & statistics

`aggregate()` groups by `(pca_label, samples_per_class)` and averages over the seeds:

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

The notebook plots the **net** value. Typical magnitude here: ≈ 0.04 J/image net (≈ 40 mJ),
vs ≈ 0.19 J/image gross.

---

## 6. Charge used per run — mAh (§9)

Charge (mAh) is the intuitive battery unit. Because the rail is a near-constant 5 V, charge
converts directly from energy:

```
Q [mAh] = E [J] / (V × 3.6)        1 mAh = 3.6 C = V × 3.6 J = 18 J  (at 5 V)
```

The notebook plots the **charge consumed per run** (one run = the ~200-image inference stream)
vs samples-per-class for each pipeline, both **gross** (idle included) and **net** (marginal,
idle removed). Typical run cost: ~**2.1 mAh gross**, ~**0.49 mAh net**. mAh figures are at the
**5 V** rail. We make **no battery-capacity assumption** — divide any pack's mAh rating by these
per-run figures to get runtime.

The per-run charge curve also surfaces a finding: **net charge/run is flat for PCA(31) across
sample sizes but climbs ~20 % for no-PCA** (160-D KNN search costs more as the store grows).

---

## 7. Exports & adding competitor models (§10)

The notebook writes model-tagged tables and the standard figures to `exports/`:

```
exports/
  experiment_summary_knn.csv        # tidy table, with a `model` column
  aggregated_by_config_knn.csv      # mean ± CI per (pca, samples), with `model`
  figures/*.png                     # accuracy / energy / memory / cpu / pca-variance
```

To add **SVM / K-means** later: produce their runs through the pipeline, then write
`experiment_summary_<model>.csv` / `aggregated_by_config_<model>.csv` into `exports/` using the
**same schema** with `model='svm'` / `model='kmeans'`. `load_all_models()` concatenates every
`experiment_summary_*.csv` so the paper's curves can be regrouped by `model` (the three-line
KNN/SVM/K-means comparison) instead of by `pca_label`.

---

## 8. Notebook section map

| § | What |
|---|---|
| 1–2 | load tidy data, aggregate across seeds |
| 3 | paper curves: accuracy / net-energy / memory / CPU vs samples (PCA vs no-PCA, CI bands) |
| 4 | sweet spot + paired PCA effect |
| 5 | seed variability |
| 6 | per-experiment deep dive (latency / accuracy / confidence timeline + confusion matrix) |
| 7 | raw power trace for one experiment |
| **8** | **KNN store size + PCA explained variance** |
| **9** | **energy as charge used per run (mAh)** + power findings |
| **10** | **export model-tagged CSVs/figures for SVM / K-means** |

## 9. Reproduce

```bash
# headless: regenerate CSVs, figures, analysis_report.md
python analyze_benchmark.py
# or interactively, run benchmark_analysis.ipynb top-to-bottom (re-crawls results/ live)
```
