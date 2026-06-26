# KNN Sweet-Spot Benchmark — Findings (CIFAR-10)

**Project:** Applied Machine Learning, DBE SS26 — *Analysis & identification of the sweet spot
for the KNN sample size in the TinyMLOps framework.*
**Hardware:** Raspberry Pi Zero 2W · external power via Voltcraft DSP-3005 (5 V rail).
**Run:** 240 experiments — pipelines `nopca` and `pca31`, 12 sample sizes (2…100 / class),
10 seeds each, 200-image CIFAR-10 **test split** (novelty) per run.

> Figures and numbers are generated from the final run by
> [`analyze_benchmark.py`](analyze_benchmark.py) / [`benchmark_analysis.ipynb`](benchmark_analysis.ipynb).
> See [`README.md`](README.md) for the measurement methodology (incl. the energy/charge derivation).

---

## TL;DR — three headline findings

1. **Diminishing returns are steep.** The first ~10 reference samples/class buy ~95% of the
   attainable accuracy; everything beyond ~30 is within noise.
2. **PCA is worth it.** 160→31 dimensions keeps **95.1%** of the variance, *slightly improves*
   accuracy (statistically significant), and cuts reference-set memory **5.2×** — at no energy or
   CPU cost.
3. **Inference is energy-cheap.** A full ~200-image run costs ~**0.49 mAh** net (~**2.2 mAh**
   gross @ 5 V); the Pi's idle draw, not the classifier, dominates power.

---

## Goal (i): KNN sample-size sweet spot

![Accuracy vs reference-sample count](figures/fig1_accuracy_vs_samples.png)

The accuracy curve is a textbook **knee**: accuracy jumps from ~53% at 2 samples/class to ~75%
at 10, then flattens.

| samples/class | 2 | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| nopca | 0.53 | 0.71 | 0.75 | 0.76 | 0.77 | 0.78 | 0.78 |
| pca31 | 0.55 | 0.72 | 0.76 | 0.76 | 0.76 | 0.78 | 0.79 |

- By **10 samples/class** the pipelines already reach **94.8% (nopca) / 95.7% (pca31)** of their
  maximum mean accuracy (max = 0.789 / 0.794).
- The strict "99% of max" knee lands at **40 samples/class (pca31)** / **60 (nopca)**, but the
  gain from 20→100 is only ~2–3 percentage points.

**Recommendation:** **~10–20 reference samples/class** is the practical sweet spot — ~95% of the
ceiling at a fraction of the memory. Use ~40 only if the last ~1% matters and memory allows.
This directly supports the on-device-retraining motivation: hoarding samples is unnecessary.

---

## Goal (ii): Usefulness of PCA

PCA(31) retains **95.14%** of the 160-D feature variance:

![PCA explained variance](figures/fig3_pca_explained_variance.png)

**Accuracy.** Paired comparison at matched (samples, seed): PCA *beats* no-PCA by **+0.70 pp** on
average (75.6% vs 74.9%), **Wilcoxon p = 0.0001** over 120 pairs — a small but real gain (likely
denoising from dropping low-variance dimensions).

**Memory — the decisive win.** Reference-store size = `samples/class × 10 classes × dim × 4 B`.
PCA shrinks the per-sample dimension 160→31, cutting the store **5.2×** (e.g. **625 KB → 121 KB**
at 100 samples/class):

![KNN store size](figures/fig2_knn_store_size.png)


---

## Power & energy

![Charge consumed per run](figures/fig4_charge_per_run.png)

- **Inference is cheap; idle dominates.** Gross draw ≈ **0.79 W**, of which only ≈ **0.18 W**
  (net) is the classifier — the rest is the Pi idling. A ~200-image run costs ≈ **2.2 mAh gross**
  but only ≈ **0.49 mAh net** (@ 5 V). *For the rover, the energy bottleneck is duty-cycle/idle,
  not the model.*
- **PCA keeps energy flat as the store grows.** Net charge/run is ~constant for PCA(31) across
  sample sizes, while no-PCA (160-D) drifts up ~**20%** from the smallest to the largest store —
  the higher-dimensional KNN search does more work. ⚠️ *Confirm BLAS threads are pinned before
  over-reading this trend (known multi-threading confound at high dimensionality).*
- **Sample size barely moves energy** in absolute terms (hundredths of a mAh per run), so the
  sweet-spot choice can be made on **accuracy + memory**, not power.

*(mAh at the 5 V supply rail; net = idle-baseline-subtracted marginal cost. We make no assumption
about battery capacity — divide any pack's mAh rating by the per-run figures for runtime.)*

---

## Methodology & rigor

- **Reproducible & seeded:** one seed controls *both* reference-sample selection and test-image
  order; 10 seeds per cell; results are mean ± 95% CI (t-based); PCA compared **paired**
  (Wilcoxon signed-rank).
- **Trained on the full training set;** only the **initialization sample size** is varied in the
  benchmark.
- **Results stabilize fast:** accuracy 95%-CI half-width shrinks from **±3.7% at 2 samples/class
  to ±2% by ≥10** — exactly where the knee is, justifying both the multi-seed design and the
  small-sample recommendation.

## Caveats

- ~79% accuracy is the **feature-extractor + KNN ceiling** on this 10-class task; test set is a
  200-image subset per run.
- The "99% knee" metric is sensitive to noise on a flat curve — frame the sweet spot as "≥95% of
  max by ~10–20 samples", not a single magic number.
- Absolute power is dominated by the Pi idle baseline (~0.61 W of ~0.79 W); we report **net**
  (marginal) energy to isolate the model.

## Appendix — overview grid

![Summary grid: accuracy / net energy / memory / CPU vs samples](figures/summary_grid.png)

**Reproduce:** `python make_findings_figures.py` regenerates every figure above from the results;
`python analyze_benchmark.py` rebuilds the CSVs + report; or run `benchmark_analysis.ipynb`
top-to-bottom for the interactive version.
