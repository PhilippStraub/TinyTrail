#!/usr/bin/env python3
"""
Benchmark analysis pipeline for the KNN Sweet-Spot project (DBE SS26).

Crawls every results/exp_* folder, extracts one tidy row of metrics per
experiment, and reproduces the paper's five-step "Experimental Analysis":

  (i)   data alignment (clock-offset correction) + baseline subtraction
  (ii)  per-experiment EDP and Accuracy
  (iii) seed-level aggregation (mean ± sample std ± 95% CI over the 10 seeds)
  (iv)  sample-size-level aggregation: config-level mean Acc/EDP, EDP growth
        (sigma_EDP) across sample-size stages, robust saturation point s*,
        and peak memory footprint (max over stages of the per-stage mean RAM)
  (v)   suitability measure per configuration, gated by the RAM budget

It works at two levels:

  LEVEL 1  experiment_summary.csv       one row per experiment (the canonical data file)
  LEVEL 2  aggregated_by_config.csv     mean ± std ± 95% CI per (config, samples_per_class),
                                        averaged over seeds (the replicates)
  LEVEL 3  suitability_by_config.csv    one row per configuration (step iv/v)
  LEVEL 3  suitability_table.csv        the same, formatted for the paper's LaTeX table

and writes, per dataset only (no more per-experiment figures):

  figures/  <dataset>_<metric>_vs_samples.pdf   individual curves with a sample-std band
            <dataset>_summary_grid.pdf          the paper figure (Springer LNCS ready)
            <dataset>_pca_explained_variance.pdf
  analysis_report.md   descriptive analysis, sweet-spot, PCA effect, methods

Usage (repo venv python):
  python analyze_benchmark.py
  python analyze_benchmark.py --results-dir ../results --out-dir ../results/analysis
  python analyze_benchmark.py --ram-budget-mb 512
"""

import argparse
import json
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
if "ipykernel" not in sys.modules and not os.environ.get("JPY_PARENT_PID"):
    matplotlib.use("Agg")       # headless CLI; keep notebook backends interactive/inline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

CLASSIFIER_LABELS = {
    "LightweightKNN": "KNN",
    "NearestCentroidClassifier": "Nearest Centroid",
    "KMeansPrototypeClassifier": "KMeans",
}

# Publication color scheme: one dark/light hue pair per classifier family
# (dark = without PCA, light = with PCA), colorblind-safe ColorBrewer tones.
# KNN is fixed to dark/light blue per the paper; the other families get their
# own hue so all configurations stay visually distinguishable in one figure.
_CLASSIFIER_COLORS: Dict[str, Tuple[str, str]] = {
    "KNN":               ("#08519C", "#9ECAE1"),  # dark blue / light blue
    "Nearest Centroid":  ("#238B45", "#A1D99B"),  # dark green / light green
    "KMeans":            ("#D94801", "#FDAE6B"),  # dark orange / light orange
}
_DEFAULT_COLORS = ("#3F3F3F", "#BFBFBF")  # dark/light gray fallback

# Short display names matching the paper's table notation (k-NN, NC, K-Means) —
# used only for the space-constrained suitability table.
_CLASSIFIER_SHORT_NAMES: Dict[str, str] = {
    "KNN": "k-NN",
    "Nearest Centroid": "NC",
    "KMeans": "K-Means",
}

# Full names for figure legends, where there is room to spell "NC" out.
_CLASSIFIER_LEGEND_NAMES: Dict[str, str] = {
    "KNN": "k-NN",
    "Nearest Centroid": "Nearest Centroid",
    "KMeans": "K-Means",
}

# The paper's 12 swept sample-size stages (stored reference samples per class).
SAMPLE_SIZES_CANONICAL = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def config_color(classifier_label: str, pca_enabled: bool) -> str:
    """Dark (no PCA) / light (PCA) hue for a given classifier family."""
    dark, light = _CLASSIFIER_COLORS.get(classifier_label, _DEFAULT_COLORS)
    return light if pca_enabled else dark


def config_display_name(classifier_label: str, pca_enabled: bool) -> str:
    """Short paper-style config label, e.g. 'k-NN + PCA', 'NC', 'K-Means' —
    for the space-constrained suitability table only."""
    short = _CLASSIFIER_SHORT_NAMES.get(classifier_label, classifier_label)
    return f"{short} + PCA" if pca_enabled else short


def config_legend_name(classifier_label: str, pca_enabled: bool) -> str:
    """Full config label for figure legends, e.g. 'Nearest Centroid + PCA'."""
    full = _CLASSIFIER_LEGEND_NAMES.get(classifier_label, classifier_label)
    return f"{full} + PCA" if pca_enabled else full

# Step (i): a PC<->Pi clock disagreement is only realigned once it reaches one
# energy-sampling period (8 Hz -> 0.125 s) — below that, "correcting" the window
# would just inject sampling-rate noise rather than remove a real skew.
CLOCK_OFFSET_REALIGN_THRESHOLD_S = 0.125

# Runs excluded from every aggregate. Was populated with 3 IDs from the pre-fix legacy
# campaign (byte-identical duplicate ZIPs misattributed to the wrong experiment) — that
# data has since been discarded and the underlying bug fixed (see
# MESSMETHODIK_ZEITSYNCHRONISATION.md §5.1 / TESTPLAN_HARDWARE_VALIDIERUNG.md §8). Stays
# empty for the current campaign; the result-attribution fix makes this failure mode
# structurally impossible, confirmed live during the hardware validation run (§7.2).
EXCLUDED_EXPERIMENTS = set()


# ──────────────────────────────────────────────────────────────────────────────
# Level 1 — extract one tidy row per experiment
# ──────────────────────────────────────────────────────────────────────────────

def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV column to boolean whether it arrived as bool or 'True'/'False'."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1"])


def _latest(path_glob) -> Optional[Path]:
    """Newest matching file by modification time (defends against leftover files
    from a previous run lingering in the same experiment folder)."""
    files = list(path_glob)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def classifier_label(classifier_type: Optional[str]) -> str:
    """Short human-readable classifier name used in multi-scenario plots."""
    if not classifier_type:
        return "unknown"
    return CLASSIFIER_LABELS.get(classifier_type, classifier_type)


def _infer_dataset_name(scenario_name: Optional[str]) -> Optional[str]:
    if not scenario_name:
        return None
    lower = scenario_name.lower()
    if "cifar10" in lower:
        return "cifar10"
    if "imagenette" in lower:
        return "imagenette"
    if "imagenet" in lower:
        return "imagenet"
    return None


def _is_excluded(exp_dir: Path) -> bool:
    """Match an experiment against EXCLUDED_EXPERIMENTS as '<scenario>/<exp_name>'."""
    return f"{exp_dir.parent.name}/{exp_dir.name}" in EXCLUDED_EXPERIMENTS


def _discover_experiment_dirs(results_dir: Path) -> List[Path]:
    """Find completed-run candidates in flat and scenario-separated result layouts.

    Drops EXCLUDED_EXPERIMENTS, so every downstream aggregate, figure and report
    rests on attributable runs only.
    """
    candidates = list(results_dir.glob("exp_*")) + list(results_dir.glob("*/exp_*"))
    dirs = sorted({p.resolve() for p in candidates if p.is_dir()})
    kept = [p for p in dirs if not _is_excluded(p)]
    dropped = len(dirs) - len(kept)
    if dropped:
        print(f"[analysis] excluded {dropped} misattributed run(s) "
              f"(see EXCLUDED_EXPERIMENTS)")
    return kept


def _read_label_count(meta: Dict, df: pd.DataFrame) -> int:
    """Best-effort class count for CIFAR-10 and Imagenette/ImageNet scenarios."""
    search_dirs = []
    for key in ("python_example_dir", "raw_data_dir"):
        if meta.get(key):
            search_dirs.append(Path(meta[key]))
    if meta.get("config_source"):
        search_dirs.append(Path(meta["config_source"]).parent)

    for base in search_dirs:
        label_path = base / "model" / "labels_object.txt"
        if not label_path.exists():
            continue
        labels = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
        if labels:
            return len(labels)

    values = []
    for col in ("ground_truth", "prediction"):
        if col in df.columns:
            values.extend(df[col].dropna().astype(str).unique().tolist())
    if values:
        return len(set(values))
    return len(CLASS_LABELS)


def _npz_int(npz, key: str) -> Optional[int]:
    if key not in npz.files:
        return None
    try:
        return int(np.asarray(npz[key]).item())
    except Exception:
        return None


def _feature_processor_candidates(meta: Dict) -> List[Path]:
    """Likely feature_processor files, ordered from exact PCA label to generic."""
    bases = []
    for key in ("python_example_dir", "raw_data_dir"):
        if meta.get(key):
            bases.append(Path(meta[key]) / "model")
    if meta.get("config_source"):
        bases.append(Path(meta["config_source"]).parent / "model")

    names = []
    pca_label = meta.get("pca_label")
    if meta.get("pca_enabled") and pca_label:
        names.extend([f"feature_processor_{pca_label}.npz", "feature_processor_pca95var.npz"])
    names.append("feature_processor.npz")

    out: List[Path] = []
    seen = set()
    for base in bases:
        for name in names:
            cand = (base / name).resolve()
            if cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
    return out


def _infer_feature_dims(meta: Dict) -> Dict[str, Optional[int]]:
    """Infer raw and stored feature dimensions from cached feature_processor .npz files."""
    raw_dim = None
    output_dim = None
    for cand in _feature_processor_candidates(meta):
        if not cand.exists():
            continue
        try:
            with np.load(cand, allow_pickle=True) as z:
                raw_dim = raw_dim or _npz_int(z, "input_dim")
                output_dim = output_dim or _npz_int(z, "output_dim")
                if raw_dim and output_dim:
                    break
        except Exception:
            continue

    meta_components = meta.get("pca_components")
    try:
        meta_components = int(meta_components) if meta_components is not None else None
    except (TypeError, ValueError):
        meta_components = None

    if meta.get("pca_enabled"):
        feature_dim = meta_components or output_dim
        pca_components = feature_dim
    else:
        feature_dim = raw_dim
        pca_components = None

    return {
        "raw_feature_dim": raw_dim,
        "feature_dim": feature_dim,
        "pca_components": pca_components,
        "pca_components_metadata": meta_components,
    }


def _read_classifier_config(meta: Dict) -> Dict:
    cfg_path = meta.get("config_source")
    if not cfg_path:
        return {}
    path = Path(cfg_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _adaptive_classifier_config(meta: Dict) -> Dict:
    cfg = _read_classifier_config(meta)
    if "tinylcm" in cfg:
        return (cfg.get("tinylcm") or {}).get("adaptive_classifier") or {}
    return cfg.get("adaptive_classifier") or {}


def _classifier_state_kb(meta: Dict, n_classes: int,
                         feature_dim: Optional[int],
                         samples_per_class: Optional[int]) -> float:
    """Approximate on-device classifier state for the vectors stored at runtime."""
    if not feature_dim or not n_classes:
        return np.nan

    ctype = meta.get("classifier_type")
    if ctype == "NearestCentroidClassifier":
        n_vectors = n_classes
    elif ctype == "KMeansPrototypeClassifier":
        acfg = _adaptive_classifier_config(meta)
        n_vectors = n_classes * int(acfg.get("m_prototypes", 3))
    else:
        if not samples_per_class:
            return np.nan
        n_vectors = n_classes * int(samples_per_class)
    return n_vectors * int(feature_dim) * 4 / 1024.0


def _confusion_counts(df: pd.DataFrame) -> Optional[Dict]:
    """Per-class one-vs-rest TP/FP/FN/TN + macro P/R/F1 from ground_truth/prediction.

    Pure post-processing over columns the Pi already writes into merged_analysis_*.csv
    (main.py's PerformanceLogger.log_inference) — no change to the on-device runtime or
    CSV schema. Rows missing either label are dropped first, mirroring the existing
    accuracy exclusion rule for unlabeled images (load_experiment's "correct" fallback).
    """
    if "ground_truth" not in df.columns or "prediction" not in df.columns:
        return None
    labeled = df.dropna(subset=["ground_truth", "prediction"])
    if labeled.empty:
        return None
    labels = sorted(set(labeled["ground_truth"]) | set(labeled["prediction"]))
    cm = pd.crosstab(labeled["ground_truth"], labeled["prediction"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    total = int(cm.values.sum())
    per_class: Dict[str, Dict] = {}
    precisions, recalls, f1s = [], [], []
    tp_sum = fp_sum = fn_sum = tn_sum = 0
    for c in labels:
        tp = int(cm.loc[c, c])
        fp = int(cm[c].sum() - tp)          # predicted c, actually something else
        fn = int(cm.loc[c].sum() - tp)      # actually c, predicted something else
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0
              else np.nan)
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": precision, "recall": recall, "f1": f1,
                        "support": int(cm.loc[c].sum())}
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn
        if pd.notna(precision):
            precisions.append(precision)
        if pd.notna(recall):
            recalls.append(recall)
        if pd.notna(f1):
            f1s.append(f1)
    return {
        "labels": labels,
        "per_class": per_class,
        "tp_micro": tp_sum, "fp_micro": fp_sum, "fn_micro": fn_sum, "tn_micro": tn_sum,
        "precision_macro": float(np.mean(precisions)) if precisions else np.nan,
        "recall_macro": float(np.mean(recalls)) if recalls else np.nan,
        "f1_macro": float(np.mean(f1s)) if f1s else np.nan,
        "n_labeled": int(len(labeled)),
    }


def _load_baseline_blocks(scenario_dir: Path) -> List[Dict]:
    """All baseline_NNN/baseline_summary.json in a scenario dir, sorted by time.

    Written by benchmark_pipeline.py's run_baseline_block() — one warm-idle block
    (n deployments, application.baseline_mode=true) before the first sample-size
    cell and after every cell. See _local_baseline_power_w() for how these are
    matched to a given experiment's time window.
    """
    blocks: List[Dict] = []
    for d in sorted(scenario_dir.glob("baseline_*")):
        summary_path = d / "baseline_summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("avg_power_W_mean") is not None:
            blocks.append(data)
    blocks.sort(key=lambda b: b.get("recorded_at", ""))
    return blocks


def _bracketing_blocks(
    blocks: List[Dict],
    started_at_iso: Optional[str],
    result_received_at_iso: Optional[str],
) -> List[Dict]:
    """The baseline block(s) immediately bracketing this experiment's time window —
    the "Mittelwert der beiden Randbloecke" design: one warm-idle measurement right
    before the run started, one right after it finished. Shared by
    _local_baseline_power_w() and _local_baseline_sem_w() so both always agree on
    which blocks a given experiment falls between.
    """
    if not blocks or not started_at_iso:
        return []
    end_iso = result_received_at_iso or started_at_iso
    before = [b for b in blocks if b.get("recorded_at", "") <= started_at_iso]
    after  = [b for b in blocks if b.get("recorded_at", "") >= end_iso]
    matched = []
    if before:
        matched.append(before[-1])
    if after:
        matched.append(after[0])
    return matched


def _local_baseline_power_w(
    blocks: List[Dict],
    started_at_iso: Optional[str],
    result_received_at_iso: Optional[str],
    fallback: Optional[float],
) -> Optional[float]:
    """Mean of the baseline block(s) immediately bracketing this experiment's time
    window. Returns `fallback` (always None from the current call site) if no
    bracketing block is found, e.g. for result sets collected before baseline
    blocks existed — there is no global config value to fall back to anymore
    (see PROMPT_BASELINE_CLEANUP.md).
    """
    matched = _bracketing_blocks(blocks, started_at_iso, result_received_at_iso)
    if not matched:
        return fallback
    return float(np.mean([b["avg_power_W_mean"] for b in matched]))


def _local_baseline_sem_w(
    blocks: List[Dict],
    started_at_iso: Optional[str],
    result_received_at_iso: Optional[str],
) -> Optional[float]:
    """Systematic (Typ-B) uncertainty of the local baseline correction above: the
    standard error of the bracketing-block mean, propagated from each block's own
    n=3 raw avg_power_W_values (see run_baseline_block() in benchmark_pipeline.py).

    This is constant across every seed of a cell — all seeds share the same
    bracketing blocks — unlike the seed-to-seed spread that _ci95() measures. The
    two are complementary, not interchangeable: _ci95() shrinks with more seeds,
    this does not (it only shrinks with more/better baseline deployments).

    SEM_lokal = sqrt(SEM_before^2 + SEM_after^2) / 2  (variance of the mean of two
    independent estimates), or SEM_block directly when only one block brackets the
    cell (campaign edge, nothing to average).
    """
    matched = _bracketing_blocks(blocks, started_at_iso, result_received_at_iso)
    sems = []
    for b in matched:
        vals = [v for v in (b.get("avg_power_W_values") or []) if v is not None]
        if len(vals) >= 2:
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
    if not sems:
        return None
    if len(sems) == 1:
        return sems[0]
    return float(np.sqrt(sum(s ** 2 for s in sems)) / 2.0)


@lru_cache(maxsize=None)
def _load_cpu_ram_baseline_blocks(scenario_dir: Path) -> List[Dict]:
    """CPU/RAM baseline per baseline_NNN block, mirroring _load_baseline_blocks()
    for power. baseline_summary.json only records power, so CPU/RAM must be read
    from each of the block's 3 warm-idle run dirs (baseline_NNN_run1/2/3) directly
    — same merged_analysis_*.csv schema an experiment uses, measured the same way
    (drop the first CPU sample as the model-load spike, mean RSS) — then averaged
    into the one three-measurement-block reading the paper's step (i) describes.

    Cached per scenario dir: every experiment in a scenario shares the same
    handful of baseline blocks, so without caching this would re-parse the same
    merged_analysis_*.csv files once per experiment instead of once per campaign.
    """
    blocks: List[Dict] = []
    for d in sorted(scenario_dir.glob("baseline_*")):
        summary_path = d / "baseline_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        recorded_at = summary.get("recorded_at")
        if not recorded_at:
            continue
        cpu_vals, rss_vals = [], []
        for run_dir in sorted(d.glob(f"{d.name}_run*")):
            merged_path = _latest(run_dir.glob("merged_analysis_*.csv"))
            if merged_path is None:
                continue
            try:
                run_df = pd.read_csv(merged_path)
            except Exception:
                continue
            if "cpu_percent_total" in run_df.columns:
                cpu = pd.to_numeric(run_df["cpu_percent_total"], errors="coerce").dropna()
                if len(cpu):
                    cpu_vals.append(float(cpu.iloc[1:].mean() if len(cpu) > 1 else cpu.mean()))
            if "process_rss_mb_resource" in run_df.columns:
                rss = pd.to_numeric(run_df["process_rss_mb_resource"], errors="coerce").dropna()
                if len(rss):
                    rss_vals.append(float(rss.mean()))
        if cpu_vals or rss_vals:
            blocks.append({
                "recorded_at": recorded_at,
                "cpu_percent_mean": float(np.mean(cpu_vals)) if cpu_vals else None,
                "memory_rss_mb_mean": float(np.mean(rss_vals)) if rss_vals else None,
            })
    blocks.sort(key=lambda b: b.get("recorded_at", ""))
    return blocks


def _local_baseline_cpu_ram(
    blocks: List[Dict],
    started_at_iso: Optional[str],
    result_received_at_iso: Optional[str],
) -> Dict[str, Optional[float]]:
    """Mean of the baseline block(s) bracketing this experiment's time window,
    for CPU and RAM — the same "average of the two bracketing blocks" rule as
    _local_baseline_power_w(), applied to _load_cpu_ram_baseline_blocks()'s output."""
    matched = _bracketing_blocks(blocks, started_at_iso, result_received_at_iso)
    if not matched:
        return {"cpu_percent": None, "memory_rss_mb": None}
    cpu_vals = [b["cpu_percent_mean"] for b in matched if b.get("cpu_percent_mean") is not None]
    rss_vals = [b["memory_rss_mb_mean"] for b in matched if b.get("memory_rss_mb_mean") is not None]
    return {
        "cpu_percent": float(np.mean(cpu_vals)) if cpu_vals else None,
        "memory_rss_mb": float(np.mean(rss_vals)) if rss_vals else None,
    }


def _offset_corrected_power_metrics(exp_dir: Path, meta: Dict, n_images: Optional[int]) -> Dict:
    """Re-integrate the raw power_*.csv over the inference window, shifted by
    the measured PC<->Pi clock offset (T_pc_equiv = T_pi - offset_s) instead of
    trusting the two clocks to already agree — see PLAN_ZEIT_HANDSHAKE.md and
    MESSMETHODIK_ZEITSYNCHRONISATION.md for the NTP-style handshake and sign
    convention. Falls back to {} (caller keeps the uncorrected
    meta["power"]["inference_window"]) when no clock_offset was measured for
    this run, e.g. data collected before this feature existed.
    """
    offset = (meta.get("clock_offset") or {}).get("offset_s")
    if offset is None:
        return {}

    res_csv = next(iter(exp_dir.glob("resource_usage_*.csv")), None)
    power_csv = next(iter(exp_dir.glob("power_*.csv")), None)
    if res_csv is None or power_csv is None:
        return {}

    import csv as _csv
    pi_times = []
    with open(res_csv, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                pi_times.append(float(row["timestamp_unix"]))
            except (ValueError, KeyError):
                continue
    if not pi_times:
        return {}
    t0_pc = min(pi_times) - offset
    t1_pc = max(pi_times) - offset

    ts, pw = _read_power_csv(power_csv)
    sel = [(t, p) for t, p in zip(ts, pw) if t0_pc <= t <= t1_pc]
    if len(sel) < 2:
        return {}
    energy = sum((pa + pb) / 2.0 * (tb - ta)
                 for (ta, pa), (tb, pb) in zip(sel, sel[1:]))
    duration = sel[-1][0] - sel[0][0]
    result = {
        "energy_J_offset_corrected": round(energy, 4),
        "avg_power_W_offset_corrected": round(energy / duration, 4) if duration > 0 else None,
        "duration_s_offset_corrected": round(duration, 3),
        "offset_applied_s": offset,
    }
    if n_images:
        result["energy_per_image_J_offset_corrected"] = round(energy / n_images, 6)
    return result


def load_experiment(exp_dir: Path) -> Optional[Dict]:
    """Return one flat dict of metrics for a single experiment, or None if incomplete."""
    meta_path = exp_dir / "experiment_metadata.json"
    merged_path = _latest(exp_dir.glob("merged_analysis_*.csv"))
    if not meta_path.exists() or merged_path is None:
        return None  # experiment didn't finish (TIMEOUT) — skip

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_csv(merged_path)
    scenario_name = meta.get("scenario_name")
    if not scenario_name and exp_dir.parent.name.startswith("scenario_"):
        scenario_name = exp_dir.parent.name
    dataset_name = meta.get("dataset_name") or _infer_dataset_name(scenario_name)
    classifier_type = meta.get("classifier_type")
    dims = _infer_feature_dims(meta)
    n_classes = _read_label_count(meta, df)
    samples_per_class = meta.get("samples_per_class")
    experiment_id = meta.get("experiment_id", exp_dir.name)
    run_id = f"{scenario_name}/{experiment_id}" if scenario_name else experiment_id

    row: Dict = {
        "run_id":            run_id,
        "experiment_id":     experiment_id,
        "scenario_name":     scenario_name,
        "dataset_name":      dataset_name,
        "classifier_type":   classifier_type,
        "classifier_label":  classifier_label(classifier_type),
        "inference_resolution": "x".join(map(str, meta.get("inference_resolution", []))) or None,
        "result_dir":        str(exp_dir),
        "pca_enabled":       meta.get("pca_enabled"),
        "pca_components":    dims["pca_components"],
        "pca_components_metadata": dims["pca_components_metadata"],
        "pca_label":         meta.get("pca_label", "pca" if meta.get("pca_enabled") else "nopca"),
        "pca_target":        meta.get("pca_target"),
        "pca_target_type":   meta.get("pca_target_type"),
        "samples_per_class": samples_per_class,
        "seed":              meta.get("seed"),
        "pca_explained_variance_pct": meta.get("pca_explained_variance_%"),
        "raw_feature_dim":   dims["raw_feature_dim"],
        "feature_dim":       dims["feature_dim"],
        "n_classes":         n_classes,
        "n_images":          int(len(df)),
    }

    # ── Accuracy ──────────────────────────────────────────────────────────
    if "final_accuracy" in df.columns and df["final_accuracy"].notna().any():
        row["accuracy"] = float(df["final_accuracy"].dropna().iloc[-1])
    elif "correct" in df.columns:
        row["accuracy"] = float(_to_bool(df["correct"]).mean())
    else:
        row["accuracy"] = np.nan

    # ── Confusion matrix / TP,FP,TN,FN ──────────────────────────────────────
    conf = _confusion_counts(df)
    if conf:
        row["tp_micro"] = conf["tp_micro"]
        row["fp_micro"] = conf["fp_micro"]
        row["fn_micro"] = conf["fn_micro"]
        row["tn_micro"] = conf["tn_micro"]
        row["precision_macro"] = conf["precision_macro"]
        row["recall_macro"] = conf["recall_macro"]
        row["f1_macro"] = conf["f1_macro"]
        row["n_labeled"] = conf["n_labeled"]
    else:
        row["tp_micro"] = row["fp_micro"] = row["fn_micro"] = row["tn_micro"] = np.nan
        row["precision_macro"] = row["recall_macro"] = row["f1_macro"] = np.nan
        row["n_labeled"] = np.nan

    # ── Latency ───────────────────────────────────────────────────────────
    if "latency_ms" in df.columns:
        lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        row["latency_ms_mean"] = float(lat.mean())
        row["latency_ms_p95"]  = float(lat.quantile(0.95))
        row["latency_sum_ms"]  = float(lat.sum())
    else:
        row["latency_ms_mean"] = row["latency_ms_p95"] = row["latency_sum_ms"] = np.nan

    # ── CPU (system-wide, duty-cycled at the rover cadence) ─────────────────
    if "cpu_percent_total" in df.columns:
        cpu = pd.to_numeric(df["cpu_percent_total"], errors="coerce").dropna()
        # drop the very first sample (model-load spike) for a cleaner steady-state mean
        row["cpu_percent_mean"] = float(cpu.iloc[1:].mean() if len(cpu) > 1 else cpu.mean())
    else:
        row["cpu_percent_mean"] = np.nan

    # ── Memory (process RSS — dominated by Python+TFLite runtime, see report) ─
    if "process_rss_mb_resource" in df.columns:
        rss = pd.to_numeric(df["process_rss_mb_resource"], errors="coerce").dropna()
        row["memory_rss_mb_mean"] = float(rss.mean())
        row["memory_rss_mb_max"]  = float(rss.max())
    else:
        row["memory_rss_mb_mean"] = row["memory_rss_mb_max"] = np.nan

    # ── Analytical vector storage (deterministic, isolates the sample/prototype cost) ──
    # support_set_kb is the selected reference budget; classifier_state_kb is the
    # vectors the deployed classifier stores at runtime (KNN samples, centroids, prototypes).
    feat_dim = row["feature_dim"]
    if feat_dim and samples_per_class:
        row["support_set_kb"] = (int(samples_per_class) * n_classes * int(feat_dim) * 4) / 1024.0
    else:
        row["support_set_kb"] = np.nan
    row["classifier_state_kb"] = _classifier_state_kb(meta, n_classes, feat_dim, samples_per_class)
    row["knn_store_kb"] = row["support_set_kb"]  # legacy column name used by older notebooks

    # ── Drift events ────────────────────────────────────────────────────────
    if "drift_detected" in df.columns:
        row["drift_events"] = int(_to_bool(df["drift_detected"]).sum())
    else:
        row["drift_events"] = np.nan

    # ── Energy / power (from the external Voltcraft trace, via metadata) ─────
    power = meta.get("power", {}) or {}
    inf = power.get("inference_window", {}) or {}
    full = power.get("full_window", {}) or {}
    avg_power_w_raw          = inf.get("avg_power_W")
    energy_per_image_j_raw   = inf.get("energy_per_image_J")
    inference_duration_s_raw = inf.get("duration_s") or full.get("duration_s")

    # ── Step (i), data alignment: clock-offset correction (measured PC<->Pi
    # handshake, see PLAN_ZEIT_HANDSHAKE.md) ──────────────────────────────────
    # Re-integrate the raw power trace shifted by the measured offset, then only
    # adopt that realigned reading as the experiment's energy value once the
    # offset reaches CLOCK_OFFSET_REALIGN_THRESHOLD_S — exactly the paper's "if
    # the deviation is >=0.125s, the affected experiment energy values are
    # aligned by the collected offset" rule. Below threshold, the raw metadata
    # reading is kept. offset_metrics/clock_offset_s are always recorded (even
    # when unused) so the measured offset can be audited across the campaign.
    clock_offset = meta.get("clock_offset") or {}
    offset_s = clock_offset.get("offset_s")
    row["clock_offset_s"] = offset_s
    row["clock_offset_round_trip_s"] = clock_offset.get("round_trip_s")
    offset_metrics = _offset_corrected_power_metrics(exp_dir, meta, row["n_images"])
    row.update(offset_metrics)

    clock_realigned = (
        offset_s is not None
        and abs(offset_s) >= CLOCK_OFFSET_REALIGN_THRESHOLD_S
        and offset_metrics.get("avg_power_W_offset_corrected") is not None
    )
    row["clock_realigned"] = clock_realigned
    if clock_realigned:
        row["avg_power_W"] = offset_metrics["avg_power_W_offset_corrected"]
        row["energy_per_image_J"] = offset_metrics.get(
            "energy_per_image_J_offset_corrected", energy_per_image_j_raw)
        row["inference_duration_s"] = offset_metrics.get(
            "duration_s_offset_corrected", inference_duration_s_raw)
    else:
        row["avg_power_W"] = avg_power_w_raw
        row["energy_per_image_J"] = energy_per_image_j_raw
        row["inference_duration_s"] = inference_duration_s_raw
    row["avg_power_W_raw"] = avg_power_w_raw  # kept for reference/audit only

    # ── Step (i), baseline subtraction + Step (ii), EDP ──────────────────────
    # Net power is computed from the nearest warm-idle baseline_* blocks in the
    # same scenario dir (see _local_baseline_power_w) — not a single global
    # config value — applied to the (possibly clock-realigned) avg_power_W
    # above, so EDP always reflects step (i) end-to-end. EDP = P_bar * T_bar,
    # scope-matched: net power over the same window as latency_ms_mean (both
    # "compute-attributable", not diluted by the delay_ms idle time — see
    # project discussion on EDP scale-matching).
    blocks = _load_baseline_blocks(exp_dir.parent)
    local_baseline_w = _local_baseline_power_w(
        blocks, meta.get("started_at"), meta.get("result_received_at"),
        fallback=None,
    )
    row["local_baseline_power_w"] = local_baseline_w
    if local_baseline_w is not None and row["avg_power_W"] is not None:
        row["net_avg_power_W_local"] = row["avg_power_W"] - local_baseline_w
    else:
        row["net_avg_power_W_local"] = np.nan

    if pd.notna(row["net_avg_power_W_local"]) and pd.notna(row["latency_ms_mean"]):
        row["EDP"] = float(row["net_avg_power_W_local"]) * float(row["latency_ms_mean"]) / 1000.0
    else:
        row["EDP"] = np.nan

    # ── Systematic (Typ-B) baseline uncertainty — see _local_baseline_sem_w() ──
    # Constant across all seeds of this cell (same bracketing blocks); propagated
    # into EDP the same way EDP itself is built (SEM * latency instead of power *
    # latency). Kept as a separate column, never folded into EDP or its seed-CI —
    # aggregate() reports it alongside, not inside, EDP_ci95.
    local_baseline_sem_w = _local_baseline_sem_w(
        blocks, meta.get("started_at"), meta.get("result_received_at"),
    )
    row["local_baseline_sem_w"] = local_baseline_sem_w
    if local_baseline_sem_w is not None and pd.notna(row["latency_ms_mean"]):
        row["EDP_systematic_sem"] = float(local_baseline_sem_w) * float(row["latency_ms_mean"]) / 1000.0
    else:
        row["EDP_systematic_sem"] = np.nan

    # ── Step (i), baseline subtraction: CPU + RAM ────────────────────────────
    # Same bracketing-block-of-3 design as power (see _load_cpu_ram_baseline_blocks),
    # applied to the raw per-experiment cpu_percent_mean/memory_rss_mb_mean above.
    # These net_* columns are the canonical CPU/RAM metrics from here on (used by
    # aggregate(), the figures, and the peak-memory-footprint in step iv) — the
    # raw values remain in cpu_percent_mean/memory_rss_mb_mean for reference.
    cpu_ram_blocks = _load_cpu_ram_baseline_blocks(exp_dir.parent)
    baseline_cpu_ram = _local_baseline_cpu_ram(
        cpu_ram_blocks, meta.get("started_at"), meta.get("result_received_at"),
    )
    row["local_baseline_cpu_percent"] = baseline_cpu_ram.get("cpu_percent")
    row["local_baseline_memory_rss_mb"] = baseline_cpu_ram.get("memory_rss_mb")
    row["net_cpu_percent_mean"] = (
        row["cpu_percent_mean"] - baseline_cpu_ram["cpu_percent"]
        if pd.notna(row["cpu_percent_mean"]) and baseline_cpu_ram.get("cpu_percent") is not None
        else np.nan
    )
    row["net_memory_rss_mb_mean"] = (
        row["memory_rss_mb_mean"] - baseline_cpu_ram["memory_rss_mb"]
        if pd.notna(row["memory_rss_mb_mean"]) and baseline_cpu_ram.get("memory_rss_mb") is not None
        else np.nan
    )
    row["net_memory_rss_mb_max"] = (
        row["memory_rss_mb_max"] - baseline_cpu_ram["memory_rss_mb"]
        if pd.notna(row["memory_rss_mb_max"]) and baseline_cpu_ram.get("memory_rss_mb") is not None
        else np.nan
    )

    return row


def build_summary(results_dir: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in _discover_experiment_dirs(results_dir):
        r = load_experiment(exp_dir)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No completed experiments found under {results_dir}")
    sort_cols = ["dataset_name", "scenario_name", "classifier_label",
                 "pca_label", "samples_per_class", "seed"]
    sort_cols = [c for c in sort_cols if c in pd.DataFrame(rows).columns]
    df = pd.DataFrame(rows).sort_values(sort_cols)
    return df.reset_index(drop=True)


def build_confusion_table(results_dir: Path) -> pd.DataFrame:
    """Long-format table: one row per (experiment, class) with TP/FP/FN/TN.

    Independent pass over the same experiment dirs as build_summary(); re-reads each
    merged_analysis_*.csv (cheap, one-time analysis run) rather than changing
    load_experiment()'s return contract, which make_findings_figures.py and the
    notebooks depend on unchanged.
    """
    rows = []
    for exp_dir in _discover_experiment_dirs(results_dir):
        meta_path = exp_dir / "experiment_metadata.json"
        merged_path = _latest(exp_dir.glob("merged_analysis_*.csv"))
        if not meta_path.exists() or merged_path is None:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        df = pd.read_csv(merged_path)
        scenario_name = meta.get("scenario_name") or exp_dir.parent.name
        experiment_id = meta.get("experiment_id", exp_dir.name)
        run_id = f"{scenario_name}/{experiment_id}"
        conf = _confusion_counts(df)
        if not conf:
            continue
        for label, c in conf["per_class"].items():
            rows.append({
                "run_id": run_id, "scenario_name": scenario_name,
                "experiment_id": experiment_id, "label": label,
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
                "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
                "support": c["support"],
            })
    return pd.DataFrame(rows)


def build_correction_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: how large the measured PC<->Pi clock deviation was
    across all its experiments, and how much step (i)'s corrections actually
    moved the numbers — a compact audit trail for the data-alignment and
    baseline-subtraction step, independent of drilling through the per-
    experiment columns in experiment_summary.csv.
    """
    group_col = "dataset_name" if "dataset_name" in summary.columns and summary["dataset_name"].notna().any() else None
    groups = list(summary.groupby(group_col)) if group_col else [("combined", summary)]

    rows = []
    for name, g in groups:
        offs = pd.to_numeric(g.get("clock_offset_s"), errors="coerce").abs().dropna()

        realigned_mask = g["clock_realigned"] == True if "clock_realigned" in g.columns else pd.Series(False, index=g.index)
        n_realigned = int(realigned_mask.sum())
        realigned_rows = g[realigned_mask]
        if n_realigned:
            power_delta = (pd.to_numeric(realigned_rows["avg_power_W_offset_corrected"], errors="coerce")
                          - pd.to_numeric(realigned_rows["avg_power_W_raw"], errors="coerce")).abs().dropna()
        else:
            power_delta = pd.Series(dtype=float)

        baseline_power = pd.to_numeric(g.get("local_baseline_power_w"), errors="coerce").dropna()
        baseline_cpu = pd.to_numeric(g.get("local_baseline_cpu_percent"), errors="coerce").dropna()
        baseline_mem = pd.to_numeric(g.get("local_baseline_memory_rss_mb"), errors="coerce").dropna()

        rows.append({
            "dataset": name,
            "n_experiments": int(len(g)),
            "clock_offset_s_mean_abs": float(offs.mean()) if len(offs) else np.nan,
            "clock_offset_s_max_abs": float(offs.max()) if len(offs) else np.nan,
            "clock_offset_realign_threshold_s": CLOCK_OFFSET_REALIGN_THRESHOLD_S,
            "n_clock_realigned": n_realigned,
            "clock_realign_power_w_delta_mean": float(power_delta.mean()) if len(power_delta) else np.nan,
            "clock_realign_power_w_delta_max": float(power_delta.max()) if len(power_delta) else np.nan,
            "baseline_power_w_subtracted_mean": float(baseline_power.mean()) if len(baseline_power) else np.nan,
            "baseline_power_w_subtracted_max": float(baseline_power.max()) if len(baseline_power) else np.nan,
            "baseline_cpu_percent_subtracted_mean": float(baseline_cpu.mean()) if len(baseline_cpu) else np.nan,
            "baseline_cpu_percent_subtracted_max": float(baseline_cpu.max()) if len(baseline_cpu) else np.nan,
            "baseline_memory_mb_subtracted_mean": float(baseline_mem.mean()) if len(baseline_mem) else np.nan,
            "baseline_memory_mb_subtracted_max": float(baseline_mem.max()) if len(baseline_mem) else np.nan,
        })
    return pd.DataFrame(rows)


def build_latency_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: total on-device inference latency summed across
    every experiment in that dataset — all configurations, seeds, and
    sample-size stages combined. Distinct from latency_sum_ms itself, which is
    only the total for one single experiment.
    """
    group_col = "dataset_name" if "dataset_name" in summary.columns and summary["dataset_name"].notna().any() else None
    groups = list(summary.groupby(group_col)) if group_col else [("combined", summary)]

    rows = []
    for name, g in groups:
        total_ms = float(pd.to_numeric(g["latency_sum_ms"], errors="coerce").sum())
        rows.append({
            "dataset": name,
            "n_experiments": int(len(g)),
            "total_inference_latency_ms": total_ms,
            "total_inference_latency_s": total_ms / 1000.0,
            "total_inference_latency_h": total_ms / 1000.0 / 3600.0,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 — aggregate across seeds (mean ± std ± 95% CI)
# ──────────────────────────────────────────────────────────────────────────────

METRICS = [
    ("accuracy",              "Accuracy"),
    ("precision_macro",       "Precision (macro)"),
    ("recall_macro",          "Recall (macro)"),
    ("f1_macro",              "F1 (macro)"),
    ("avg_power_W",           "Avg power (W)"),
    ("net_cpu_percent_mean",     "Net CPU, local baseline (%)"),
    ("net_memory_rss_mb_mean",   "Net memory RSS, local baseline (MB)"),
    ("net_memory_rss_mb_max",    "Peak net memory RSS, local baseline (MB)"),
    ("cpu_percent_mean",      "CPU, raw (%)"),
    ("memory_rss_mb_mean",    "Memory RSS, raw (MB)"),
    ("memory_rss_mb_max",     "Peak memory RSS, raw (MB)"),
    ("classifier_state_kb",   "Classifier state (KB)"),
    ("support_set_kb",        "Reference support budget (KB)"),
    ("latency_ms_mean",       "Latency (ms)"),
    ("latency_sum_ms",        "Total inference time (ms)"),
    ("net_avg_power_W_local", "Net avg power, local baseline (W)"),
    ("EDP",                   "Net energy per inference (J)"),
]


def _ci95(x: np.ndarray) -> float:
    """Half-width of the 95% confidence interval of the mean (t-based)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    sem = np.std(x, ddof=1) / np.sqrt(n)
    if HAVE_SCIPY:
        return float(scipy_stats.t.ppf(0.975, n - 1) * sem)
    return float(1.96 * sem)  # normal approx fallback


def aggregate(summary: pd.DataFrame, group_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["pca_label", "pca_enabled", "samples_per_class"]
    group_cols = list(group_cols)
    out_rows = []
    for keys, g in summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec["n_seeds"] = int(len(g))
        for col, _ in METRICS:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
            rec[f"{col}_std"]  = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else np.nan
            rec[f"{col}_ci95"] = _ci95(vals)

        # Systematic (Typ-B) baseline uncertainty — see _local_baseline_sem_w().
        # Constant across every seed in this cell (same bracketing baseline
        # blocks), so averaging over the group is meaningless; take the one
        # distinct value instead. Reported as its own column, deliberately never
        # combined into EDP_ci95 above — the two answer different questions
        # (random seed spread vs. systematic baseline-measurement spread).
        if "EDP_systematic_sem" in g.columns:
            sem_vals = pd.to_numeric(g["EDP_systematic_sem"], errors="coerce").dropna()
            sem = float(sem_vals.iloc[0]) if len(sem_vals) else np.nan
        else:
            sem = np.nan
        rec["EDP_sem_systematic"]  = sem
        rec["EDP_ci95_systematic"] = float(1.96 * sem) if pd.notna(sem) else np.nan

        out_rows.append(rec)
    sort_cols = [c for c in group_cols if c in {"dataset_name", "scenario_name",
                                                "classifier_label", "pca_label",
                                                "samples_per_class"}]
    return pd.DataFrame(out_rows).sort_values(sort_cols).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Level 3 — paper figures
# ──────────────────────────────────────────────────────────────────────────────

# Metric -> (paper-style label with unit, is-a-percentage-in-[0,1])
_METRIC_UNITS: Dict[str, Tuple[str, bool]] = {
    "accuracy":           ("Accuracy (%)", True),
    "precision_macro":    ("Precision, macro (%)", True),
    "recall_macro":       ("Recall, macro (%)", True),
    "f1_macro":           ("F1, macro (%)", True),
    "avg_power_W":        ("Avg. power (W)", False),
    "net_cpu_percent_mean":    ("CPU utilization (%)", False),
    "net_memory_rss_mb_mean": ("Memory (MB)", False),
    "net_memory_rss_mb_max":  ("Peak memory (MB)", False),
    "cpu_percent_mean":   ("CPU utilization, raw (%)", False),
    "memory_rss_mb_mean": ("Memory, raw (MB)", False),
    "memory_rss_mb_max":  ("Peak memory, raw (MB)", False),
    "classifier_state_kb": ("Classifier state (KB)", False),
    "support_set_kb":     ("Reference support budget (KB)", False),
    "latency_ms_mean":    ("Latency (ms)", False),
    "latency_sum_ms":     ("Total inference time (ms)", False),
    "net_avg_power_W_local": ("Net avg. power (W)", False),
    "EDP":                ("Energy per inference (J)", False),
}


def _config_key(row: pd.Series, default_classifier: Optional[str]) -> Tuple[str, bool]:
    """(classifier_label, pca_enabled) for one aggregated row, robust to both
    the single-classifier (agg has no classifier_label column) and
    multi-classifier (agg has classifier_label/pca_enabled columns) shapes
    that aggregate() can produce."""
    classifier = row.get("classifier_label", default_classifier) or default_classifier or "unknown"
    pca_enabled = bool(row.get("pca_enabled", False))
    return classifier, pca_enabled


def _present_families(agg: pd.DataFrame, default_classifier: Optional[str]) -> List[str]:
    """Classifier families present in this aggregate, in a fixed, stable order
    (so the legend column order never shuffles between figures/datasets)."""
    if "classifier_label" in agg.columns:
        present = set(agg["classifier_label"].dropna().unique().tolist())
    else:
        present = {default_classifier or "unknown"}
    order = ["KNN", "KMeans", "Nearest Centroid"]
    return [f for f in order if f in present] + sorted(present - set(order))


SAMPLE_SIZE_TICKS = [2, 20, 40, 60, 80, 100]


def _set_sample_size_ticks(ax, agg: pd.DataFrame) -> None:
    """Fixed, evenly legible x-ticks (2, 20, 40, 60, 80, 100) instead of either
    matplotlib's default linear ticks (which land on unmeasured numbers like 25,
    75) or all 12 swept stages (2, 5, 10, ... crowd together at the low end and
    overlap even when rotated). Horizontal, unrotated labels."""
    ax.set_xticks(SAMPLE_SIZE_TICKS)
    ax.set_xticklabels([str(t) for t in SAMPLE_SIZE_TICKS], rotation=0, ha="center")


def _plot_metric_vs_samples(ax, agg: pd.DataFrame, metric: str, unit_label: str,
                            default_classifier: Optional[str] = None,
                            show_legend: bool = True, percent_scale: bool = False):
    """One metric vs samples-per-class, one line per configuration, with a
    sample-standard-deviation band (step iii: "mean and sample standard
    deviation across the ten seeds") — not a confidence interval.

    Colors and labels come from config_color()/config_legend_name() (dark =
    no PCA, light = PCA, one hue per classifier family) instead of the
    matplotlib default cycle, so the same configuration always gets the same
    color across every figure.
    """
    group_cols = [c for c in ("classifier_label", "pca_enabled") if c in agg.columns]
    if not group_cols:
        group_cols = ["pca_enabled"] if "pca_enabled" in agg.columns else ["pca_label"]
    scale = 100.0 if percent_scale else 1.0

    for _, sub in agg.groupby(group_cols, dropna=False):
        sub = sub.sort_values("samples_per_class")
        classifier, pca_enabled = _config_key(sub.iloc[0], default_classifier)
        color = config_color(classifier, pca_enabled)
        label = config_legend_name(classifier, pca_enabled)

        x = sub["samples_per_class"].to_numpy()
        y = sub[f"{metric}_mean"].to_numpy() * scale
        std = sub[f"{metric}_std"].to_numpy() * scale
        if np.all(np.isnan(y)):
            continue
        ax.plot(x, y, marker="o", markersize=3.5, lw=1.3, color=color, label=label)
        std_safe = np.nan_to_num(std, nan=0.0)
        ax.fill_between(x, y - std_safe, y + std_safe, alpha=0.16, color=color, linewidth=0)

    ax.set_xlabel("Reference samples per class")
    ax.set_ylabel(unit_label)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _set_sample_size_ticks(ax, agg)
    if show_legend:
        ax.legend(frameon=False, fontsize=7)


def _shared_legend(fig, agg: pd.DataFrame, default_classifier: Optional[str]) -> None:
    """One legend below the grid: one column per classifier family (KNN |
    K-Means | Nearest Centroid), no-PCA in the top row and the matching +PCA
    entry directly underneath in the same column.

    matplotlib's Legend fills column-major (down each column first, then the
    next column) — NOT row-major. So the handles must be interleaved per
    family, [dark_f1, light_f1, dark_f2, light_f2, ...], with ncol=len(families);
    a [all-dark, then all-light] handle order (which would be right under a
    row-major fill) actually scatters each pair across different columns.
    """
    families = _present_families(agg, default_classifier)
    if not families:
        return
    handles, labels = [], []
    for f in families:
        handles.append(Line2D([0], [0], color=config_color(f, False), marker="o",
                              markersize=4, lw=1.3))
        labels.append(config_legend_name(f, False))
        handles.append(Line2D([0], [0], color=config_color(f, True), marker="o",
                              markersize=4, lw=1.3))
        labels.append(config_legend_name(f, True))
    fig.legend(handles, labels, loc="lower center", ncol=len(families),
              frameon=False, bbox_to_anchor=(0.5, -0.02))


def plot_paper_curves(agg: pd.DataFrame, fig_dir: Path, dataset_tag: str,
                      default_classifier: Optional[str] = None):
    """Dataset-level figures only (no per-experiment plots).

    Every individual metric curve is kept (as requested), but only
    `<dataset>_summary_grid.pdf` is meant for the paper — it gets the extra
    Springer LNCS formatting pass (no header/suptitle, serif font sized for
    camera-ready text, tight LNCS-column figure size, % units on accuracy).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Individual figures — one metric per file, full curve set incl. legend.
    for metric, _ in METRICS:
        unit_label, is_pct = _METRIC_UNITS.get(metric, (metric, False))
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric_vs_samples(ax, agg, metric, unit_label,
                                default_classifier=default_classifier,
                                percent_scale=is_pct)
        ax.set_title(f"{dataset_tag}: {unit_label} vs. samples per class")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dataset_tag}_{metric}_vs_samples.pdf")
        plt.close(fig)

    # 2×2 paper grid (accuracy, EDP, memory, CPU) — Springer LNCS camera-ready.
    grid_metrics = ["accuracy", "EDP", "net_memory_rss_mb_mean", "latency_sum_ms"]
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    # LNCS single-column text width is 122 mm; keep the whole grid inside it
    # and use a serif face + small paper-appropriate font sizes.
    lncs_width_in = 122.0 / 25.4
    with plt.rc_context({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
    }):
        # Modest extra height for the 2-row (no-PCA / +PCA) legend below the grid.
        fig, axes = plt.subplots(2, 2, figsize=(lncs_width_in, lncs_width_in * 0.84))
        for ax, metric, tag in zip(axes.ravel(), grid_metrics, panel_tags):
            unit_label, is_pct = _METRIC_UNITS.get(metric, (metric, False))
            _plot_metric_vs_samples(ax, agg, metric, unit_label,
                                    default_classifier=default_classifier,
                                    show_legend=False, percent_scale=is_pct)
            ax.text(0.02, 0.96, tag, transform=ax.transAxes, fontsize=7,
                    fontweight="bold", va="top", ha="left")
        # Single shared legend directly under the grid instead of a header/title —
        # one column per family, no-PCA/+PCA stacked within it (see _shared_legend()).
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        _shared_legend(fig, agg, default_classifier)
        fig.savefig(fig_dir / f"{dataset_tag}_summary_grid.pdf")
        plt.close(fig)


def plot_pca_variance(summary: pd.DataFrame, fig_dir: Path, dataset_tag: str):
    sub = summary[summary["pca_enabled"] == True].dropna(subset=["pca_explained_variance_pct"])
    if sub.empty:
        return
    by_comp = sub.groupby("pca_components")["pca_explained_variance_pct"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(int(c)) for c in by_comp.index], by_comp.values,
          color=_CLASSIFIER_COLORS.get("KNN", _DEFAULT_COLORS)[1])
    for i, v in enumerate(by_comp.values):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_title(f"{dataset_tag}: PCA explained variance")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Explained variance (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{dataset_tag}_pca_explained_variance.pdf")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Power trace parsing (used by the clock-offset re-integration in step (i))
# ──────────────────────────────────────────────────────────────────────────────

def _read_power_csv(power_csv: Path):
    """Parallel lists (epoch_seconds, power_W) from a dsp3005 power log.

    The DSP timestamps are PC-local naive ISO strings; datetime.timestamp()
    converts them to true epoch seconds using the local timezone — resolution-
    independent and consistent with how the pipeline computes energy windows.
    """
    ts, pw = [], []
    p = pd.read_csv(power_csv)
    for t, v in zip(p.get("timestamp", []), p.get("P_out", [])):
        try:
            ts.append(datetime.fromisoformat(str(t)).timestamp())
            pw.append(float(v))
        except (ValueError, TypeError):
            continue
    return ts, pw


# ──────────────────────────────────────────────────────────────────────────────
# Statistics for the report — sweet spot + PCA effect
# ──────────────────────────────────────────────────────────────────────────────

def find_sweet_spot(agg: pd.DataFrame, frac: float = 0.99) -> Dict:
    """Smallest samples-per-class reaching `frac` of the max mean accuracy, per pipeline."""
    out = {}
    label_col = "plot_label" if "plot_label" in agg.columns else "pca_label"
    for label, sub in agg.groupby(label_col):
        sub = sub.sort_values("samples_per_class")
        acc = sub["accuracy_mean"]
        if acc.notna().sum() == 0:
            continue
        target = frac * acc.max()
        reached = sub[acc >= target]
        if not reached.empty:
            out[label] = {
                "max_accuracy": float(acc.max()),
                "target_accuracy": float(target),
                "sweet_spot_samples_per_class": int(reached["samples_per_class"].iloc[0]),
            }
    return out


def _robust_saturation_by_group(agg: pd.DataFrame, group_cols: Sequence[str],
                                frac: float = 0.95) -> Dict[Tuple, Dict]:
    """Core of the robust saturation point s* (step iv): smallest samples-per-class
    stage whose mean accuracy reaches `frac` of the max mean accuracy AND whose
    margin above that threshold exceeds the standard error of the mean (SEM) at
    that stage — keyed by the raw `group_cols` tuple rather than a display label,
    so callers can join it back onto any grouping of `agg` (per-label report dict
    via find_robust_saturation_point(), or per-classifier tuple via
    compute_suitability()).

    The SEM check guards against picking a stage that only clears the threshold
    due to seed noise — the same size-2 cell that happens to score high by luck.
    Uses accuracy_std/n_seeds already produced by aggregate(); a stage without
    that data (n_seeds<2) is accepted on the threshold alone, since robustness
    can't be assessed there.
    """
    out: Dict[Tuple, Dict] = {}
    for keys, sub in agg.groupby(list(group_cols), dropna=False):
        key = keys if isinstance(keys, tuple) else (keys,)
        sub = sub.sort_values("samples_per_class")
        acc = sub["accuracy_mean"]
        if acc.notna().sum() == 0:
            continue
        max_acc = float(acc.max())
        target = frac * max_acc

        s_star = None
        for _, r in sub.iterrows():
            if pd.isna(r["accuracy_mean"]) or r["accuracy_mean"] < target:
                continue
            margin = r["accuracy_mean"] - target
            std = r.get("accuracy_std")
            n_seeds = r.get("n_seeds")
            if pd.notna(std) and n_seeds and n_seeds > 1:
                sem = std / (n_seeds ** 0.5)
            else:
                sem = 0.0
            if margin > sem:
                s_star = int(r["samples_per_class"])
                break

        if s_star is not None:
            out[key] = {
                "max_accuracy": max_acc,
                "target_accuracy": float(target),
                "robust_saturation_samples_per_class": s_star,
            }
    return out


def find_robust_saturation_point(agg: pd.DataFrame, frac: float = 0.95) -> Dict:
    """Robust saturation point s*, keyed by the plot/pca display label (report use)."""
    label_col = "plot_label" if "plot_label" in agg.columns else "pca_label"
    by_group = _robust_saturation_by_group(agg, [label_col], frac=frac)
    return {key[0]: val for key, val in by_group.items()}


def _saturation_by_group(agg: pd.DataFrame, group_cols: Sequence[str],
                         frac: float = 0.95) -> Dict[Tuple, int]:
    """Robust saturation point s*, keyed by the raw group tuple (e.g.
    (classifier_label, pca_enabled)) — used by compute_suitability()."""
    by_group = _robust_saturation_by_group(agg, group_cols, frac=frac)
    return {key: val["robust_saturation_samples_per_class"] for key, val in by_group.items()}


def compute_suitability(
    agg: pd.DataFrame,
    s_min: Optional[int] = None,
    ram_budget_mb: Optional[float] = None,
    default_classifier: Optional[str] = None,
) -> pd.DataFrame:
    """Suitability_c = Acc_bar_c / (EDP_bar_c + sigma_EDP_c) * (s_min / s*_c) per
    configuration c (classifier_type x pca_label — step (iv)/(v) of the paper).

    EDP_bar_c/sigma_EDP_c are the mean/std of the already seed-averaged per-stage
    EDP (aggregate()'s "EDP_mean" column) taken ACROSS samples-per-class stages —
    sigma_EDP_c therefore captures energy-cost growth over the sample-size
    trajectory, not seed-to-seed noise (that's already averaged out per stage).

    Peak memory footprint: the average (baseline-subtracted) RAM utilization per
    stage (aggregate()'s "net_memory_rss_mb_mean_mean") taken at its worst
    sample-size stage (max over stages) — RAM grows monotonically with
    samples_per_class for KNN, so a config can look fine on average and still
    exceed the budget at the largest sample count. Used only as a feasibility
    filter against `ram_budget_mb`, never folded into the suitability score itself.
    """
    group_cols = [c for c in ("classifier_label", "pca_enabled") if c in agg.columns]
    if not group_cols:
        group_cols = ["pca_enabled"] if "pca_enabled" in agg.columns else ["pca_label"]
    if s_min is None:
        s_min = int(agg["samples_per_class"].min())

    saturation_by_group = _saturation_by_group(agg, group_cols, frac=0.95)

    rows = []
    for keys, sub in agg.groupby(group_cols, dropna=False):
        sub = sub.sort_values("samples_per_class")
        classifier, pca_enabled = _config_key(sub.iloc[0], default_classifier)
        group_key = keys if isinstance(keys, tuple) else (keys,)

        acc_bar = sub["accuracy_mean"].mean()

        edp_per_stage = sub["EDP_mean"] if "EDP_mean" in sub.columns else pd.Series(dtype=float)
        edp_valid = edp_per_stage.dropna()
        edp_bar = edp_valid.mean() if len(edp_valid) else np.nan
        sigma_edp = edp_valid.std(ddof=1) if len(edp_valid) > 1 else np.nan

        # Sanity check: is the observed stage-to-stage EDP spread (sigma_edp) real
        # signal, or largely baseline measurement floor noise? Each stage hangs off
        # a different bracketing block pair, so part of sigma_edp can come from
        # baseline-estimate imprecision rather than genuine energy growth over the
        # sample-size curve. Compare against the typical per-stage systematic SEM.
        sem_sys = sub["EDP_sem_systematic"] if "EDP_sem_systematic" in sub.columns else pd.Series(dtype=float)
        sem_sys_valid = sem_sys.dropna()
        edp_sem_systematic_typical = sem_sys_valid.mean() if len(sem_sys_valid) else np.nan
        sigma_edp_over_systematic = (
            sigma_edp / edp_sem_systematic_typical
            if pd.notna(sigma_edp) and pd.notna(edp_sem_systematic_typical) and edp_sem_systematic_typical > 0
            else np.nan
        )

        mem_col = "net_memory_rss_mb_mean_mean" if "net_memory_rss_mb_mean_mean" in sub.columns else None
        mem_feasibility = sub[mem_col].max() if mem_col else np.nan

        s_star = saturation_by_group.get(group_key)

        if pd.notna(acc_bar) and pd.notna(edp_bar) and pd.notna(sigma_edp) and s_star:
            denom = edp_bar + sigma_edp
            suitability = (acc_bar / denom) * (s_min / s_star) if denom > 0 else np.nan
        else:
            suitability = np.nan

        feasible = (
            bool(mem_feasibility <= ram_budget_mb)
            if (ram_budget_mb is not None and pd.notna(mem_feasibility))
            else None
        )

        rows.append({
            "config": config_display_name(classifier, pca_enabled),
            "classifier_label": classifier,
            "pca_enabled": pca_enabled,
            "accuracy_bar": acc_bar,
            "EDP_bar": edp_bar,
            "sigma_EDP": sigma_edp,
            "EDP_sem_systematic_typical": edp_sem_systematic_typical,
            "sigma_EDP_over_systematic": sigma_edp_over_systematic,
            "s_star": s_star,
            "s_min": s_min,
            "memory_feasibility_mb": mem_feasibility,
            "ram_budget_mb": ram_budget_mb,
            "feasible": feasible,
            "suitability": suitability,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("suitability", ascending=False, na_position="last")
    return out


def _round_sig(x: pd.Series, sig: int = 3) -> pd.Series:
    """Round to `sig` significant figures instead of a fixed decimal count.

    EDP spans two-plus orders of magnitude across datasets (~5e-3 J for
    CIFAR-10/Imagenette, ~5e-5 J for MNIST) — a fixed .round(4) reads fine for
    the former and collapses the latter to 0.0000 for almost every row. Sig-fig
    rounding keeps the same number of meaningful digits regardless of scale.
    """
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(vals, np.nan)
    finite_nonzero = np.isfinite(vals) & (vals != 0)
    magnitude = np.floor(np.log10(np.abs(vals[finite_nonzero])))
    decimals = (sig - 1 - magnitude).astype(int)
    out[finite_nonzero] = [round(v, d) for v, d in zip(vals[finite_nonzero], decimals)]
    out[np.isfinite(vals) & (vals == 0)] = 0.0
    return pd.Series(out, index=x.index)


def build_suitability_table(suitability: pd.DataFrame, dataset_tag: str) -> pd.DataFrame:
    """Format compute_suitability()'s output as the paper's LaTeX table
    (tab:suitability): Case, Config, Acc (%), EDP-bar, sigma_EDP, s*, Suit. (%),
    RAM (MB), RAM (%). Suit. and RAM (%) are relative to the best-performing
    configuration and the RAM budget, respectively — matching the table
    caption; RAM (MB) is the underlying absolute peak memory footprint (step
    iv) that RAM (%) is computed from. One CSV per dataset, but the "Case"
    column repeats the dataset name per row so the per-dataset CSVs
    concatenate directly into the combined multirow table.
    """
    if suitability.empty:
        return pd.DataFrame(columns=["Case", "Config", "Acc (%)", "EDP_bar", "sigma_EDP",
                                     "s_star", "Suit (%)", "RAM (MB)", "RAM (%)"])

    max_suit = suitability["suitability"].max()
    ram_budget = suitability["ram_budget_mb"].dropna()
    ram_budget = float(ram_budget.iloc[0]) if len(ram_budget) else np.nan

    out = pd.DataFrame({
        "Case": dataset_tag,
        "Config": suitability["config"],
        "Acc (%)": (suitability["accuracy_bar"] * 100).round(2),
        "EDP_bar": _round_sig(suitability["EDP_bar"], sig=3),
        "sigma_EDP": _round_sig(suitability["sigma_EDP"], sig=3),
        "s_star": suitability["s_star"],
        "Suit (%)": ((suitability["suitability"] / max_suit) * 100).round(1)
                    if pd.notna(max_suit) and max_suit > 0 else np.nan,
        "RAM (MB)": suitability["memory_feasibility_mb"].round(3),
        "RAM (%)": ((suitability["memory_feasibility_mb"] / ram_budget) * 100).round(3)
                   if pd.notna(ram_budget) and ram_budget > 0 else np.nan,
    })
    out = out.sort_values("Suit (%)", ascending=False, na_position="last").reset_index(drop=True)

    # No robust saturation point (step iv's SEM check never passed) leaves
    # s_star/suitability undefined for that configuration — an explicit "n/a"
    # marks this as a deliberate methodological exclusion, not a missing value.
    out["s_star"] = out["s_star"].apply(lambda v: "n/a" if pd.isna(v) else str(int(v)))
    out["Suit (%)"] = out["Suit (%)"].apply(lambda v: "n/a" if pd.isna(v) else v)
    return out


def pca_effect(summary: pd.DataFrame) -> Dict:
    """Paired comparison PCA vs no-PCA at matched (samples_per_class, seed)."""
    piv = summary.pivot_table(index=["samples_per_class", "seed"],
                              columns="pca_label", values=["accuracy", "EDP"])
    out = {}
    for metric in ["accuracy", "EDP"]:
        try:
            cols = piv[metric].columns.tolist()
        except KeyError:
            continue
        if len(cols) < 2:
            continue
        a_label = next((c for c in cols if c.startswith("pca")), cols[0])
        b_label = next((c for c in cols if c.startswith("nopca")), cols[-1])
        pair = piv[metric][[a_label, b_label]].dropna()
        if len(pair) < 2:
            continue
        diff = pair[a_label] - pair[b_label]
        rec = {
            "pca_label": a_label, "nopca_label": b_label, "n_pairs": int(len(pair)),
            "mean_pca": float(pair[a_label].mean()), "mean_nopca": float(pair[b_label].mean()),
            "mean_diff_pca_minus_nopca": float(diff.mean()),
        }
        if HAVE_SCIPY and len(pair) >= 5 and diff.abs().sum() > 0:
            try:
                w, p = scipy_stats.wilcoxon(pair[a_label], pair[b_label])
                rec["wilcoxon_p"] = float(p)
            except ValueError:
                pass
        out[metric] = rec
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _table(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def write_report(summary: pd.DataFrame, agg: pd.DataFrame,
                 sweet: Dict, pca: Dict, suitability: pd.DataFrame,
                 dataset_tag: str, out_path: Path):
    n_exp = len(summary)
    seeds = sorted(summary["seed"].dropna().unique().tolist())
    spc = sorted(summary["samples_per_class"].dropna().unique().tolist())
    pcas = summary["pca_label"].unique().tolist()
    n_images = int(summary["n_images"].median())

    lines: List[str] = []
    lines.append("# Benchmark Analysis Report\n")
    lines.append("_Auto-generated by `analyze_benchmark.py`. Re-run to refresh._\n")

    lines.append("## 1. Experimental design\n")
    lines.append(f"- Completed experiments: **{n_exp}**")
    lines.append(f"- Pipelines: {', '.join(pcas)}")
    lines.append(f"- Samples per class swept: {spc}")
    lines.append(f"- Seeds (replicates): {seeds}  (**n = {len(seeds)}** per cell)")
    lines.append(f"- Test images per run (median): {n_images}")
    pca_rows = summary[summary['pca_enabled'] == True]
    if not pca_rows.empty:
        for comp, g in pca_rows.groupby("pca_components"):
            ev = g["pca_explained_variance_pct"].mean()
            lines.append(f"- PCA({int(comp)}) explained variance: **{ev:.2f}%**")
    lines.append("")

    lines.append("## 2. Methods (averaging & statistics)\n")
    lines.append("- Each (pipeline × samples-per-class) cell is repeated over the seeds above; "
                 "the seed controls **both** reference-sample selection and test-image order.")
    lines.append("- Reported values are the **arithmetic mean across seeds**; error bands/intervals "
                 "are the **95% confidence interval of the mean** (t-distribution, n−1 df).")
    lines.append("- **Systematic vs. random uncertainty (EDP)**: the seed-CI above is a random (Typ-A) "
                 "uncertainty — it shrinks with more seeds but cannot see the baseline correction, since "
                 "every seed in a cell shares the same bracketing warm-idle baseline block pair. `EDP_ci95_"
                 "systematic` is a separate, non-shrinking systematic (Typ-B) uncertainty propagated from "
                 "the raw n=3 measurements inside those baseline blocks. The two are reported side by side "
                 "and never summed into one number; a cell-to-cell EDP difference smaller than the combined "
                 "systematic uncertainty of both cells is not robustly supported by the data alone, "
                 "regardless of how tight the seed-CI looks.")
    lines.append("- PCA vs no-PCA is compared **paired** at matched (samples, seed) using the "
                 "Wilcoxon signed-rank test where ≥5 pairs are available.")
    lines.append("- **Clock alignment**: an experiment's PC<->Pi clock offset is only realigned "
                 f"(re-integrating the raw power trace) once it reaches {CLOCK_OFFSET_REALIGN_THRESHOLD_S}s "
                 "— one energy-sampling period (8 Hz); below that, the raw metadata reading is used "
                 "unchanged (`clock_realigned` flags which rule applied per experiment).")
    lines.append("- **Energy** is the (possibly clock-realigned) external Voltcraft measurement, "
                 "baseline-subtracted against the local warm-idle blocks bracketing each cell (`EDP`) "
                 "→ marginal cost of one inference, independent of duty cycle.")
    lines.append("- **CPU% and Memory (RSS)** are likewise baseline-subtracted against the same "
                 "bracketing warm-idle blocks (`net_cpu_percent_mean`, `net_memory_rss_mb_mean`); "
                 "raw (non-baseline-subtracted) values remain available for reference. CPU% is the "
                 "system-wide, duty-cycled value at the configured rover cadence (not peak inference "
                 "CPU). Memory (RSS) includes the Python+TFLite runtime baseline it is corrected "
                 "against; the marginal store for samples is the analytical `knn_store_kb` column "
                 "(spc × classes × dim × 4 B).")
    lines.append("")

    lines.append("## 3. Sweet spot (accuracy)\n")
    if sweet:
        for label, s in sweet.items():
            lines.append(f"- **{label}**: max accuracy {s['max_accuracy']:.4f}; "
                         f"reaches 99% of it ({s['target_accuracy']:.4f}) at "
                         f"**{s['sweet_spot_samples_per_class']} samples/class**.")
    else:
        lines.append("_Not enough accuracy data to locate a knee._")
    lines.append("")

    lines.append("## 4. PCA effect (paired)\n")
    if pca:
        for metric, r in pca.items():
            line = (f"- **{metric}**: mean(PCA)={r['mean_pca']:.4f}, "
                    f"mean(no-PCA)={r['mean_nopca']:.4f}, "
                    f"Δ(PCA−noPCA)={r['mean_diff_pca_minus_nopca']:+.4f} "
                    f"over {r['n_pairs']} pairs")
            if "wilcoxon_p" in r:
                line += f"; Wilcoxon p={r['wilcoxon_p']:.4f}"
            lines.append(line)
    else:
        lines.append("_Need both PCA and no-PCA runs at matched settings for a paired test._")
    lines.append("")

    lines.append("## 5. Aggregated table (mean over seeds)\n")
    show_cols = ["pca_label", "samples_per_class", "n_seeds",
                 "accuracy_mean", "accuracy_ci95",
                 "net_cpu_percent_mean_mean",
                 "net_memory_rss_mb_mean_mean", "latency_ms_mean_mean",
                 "EDP_mean", "EDP_ci95", "EDP_ci95_systematic"]
    show_cols = [c for c in show_cols if c in agg.columns]
    lines.append(_table(agg[show_cols].round(4)))
    lines.append("")

    lines.append("## 6. Suitability per configuration (step v)\n")
    lines.append("- Suitability_c = Acc_bar_c / (EDP_bar_c + sigma_EDP_c) * (s_min / s*_c), "
                 "s*_c the robust saturation point (step iv); `feasible` gates on the peak "
                 "per-stage average RAM utilization against the configured RAM budget.")
    if not suitability.empty:
        show_cols_suit = ["config", "accuracy_bar", "EDP_bar", "sigma_EDP", "s_star",
                          "memory_feasibility_mb", "feasible", "suitability"]
        show_cols_suit = [c for c in show_cols_suit if c in suitability.columns]
        lines.append(_table(suitability[show_cols_suit].round(4)))
    else:
        lines.append("_Not enough sample-size stages to compute a suitability score._")
    lines.append("")

    lines.append("## 7. Figures\n")
    lines.append(f"- `figures/{dataset_tag}_summary_grid.pdf` — the paper figure "
                 "(2×2: accuracy/EDP/memory/CPU vs samples, Springer LNCS formatted)")
    lines.append(f"- `figures/{dataset_tag}_<metric>_vs_samples.pdf` — individual curves with a sample-std band")
    lines.append(f"- `figures/{dataset_tag}_pca_explained_variance.pdf`")
    lines.append("")

    lines.append("## 8. Suitability table (LaTeX-ready)\n")
    lines.append("- `suitability_table.csv` — Case/Config/Acc (%)/EDP_bar/sigma_EDP/s_star/"
                 "Suit (%)/RAM (%), ready to be typeset as the paper's suitability table.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _analyze_group(
    summary: pd.DataFrame,
    confusion_full: pd.DataFrame,
    out_dir: Path,
    ram_budget_mb: Optional[float],
    tag: Optional[str] = None,
) -> None:
    """Run the aggregate/plot/report pipeline for one slice of `summary` (either the
    whole campaign, or a single dataset group's rows), writing everything under
    `out_dir`. Split out of main() so a multi-dataset campaign can produce one
    self-contained figures+report bundle per dataset instead of one plot with a
    line per (scenario x pca) combination — see main()'s per-dataset-group loop.
    Figures are dataset-level only; `tag` (the dataset name) is stamped into
    every figure filename.
    """
    prefix = f"[analyze:{tag}]" if tag else "[analyze]"
    dataset_tag = tag or "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_scenario = summary["scenario_name"].nunique(dropna=True) > 1
    if multi_scenario:
        group_cols = ["scenario_name", "classifier_label", "pca_label",
                      "pca_enabled", "samples_per_class"]
    else:
        group_cols = None
    agg = aggregate(summary, group_cols)
    if multi_scenario:
        label_col = "classifier_label" if summary["classifier_label"].nunique(dropna=True) > 1 else "scenario_name"
        agg["plot_label"] = agg[label_col].fillna("unknown") + " / " + agg["pca_label"].fillna("unknown")
    agg.to_csv(out_dir / "aggregated_by_config.csv", index=False)
    print(f"{prefix} {len(agg)} (pca x samples) cells -> aggregated_by_config.csv")

    if not confusion_full.empty:
        confusion = confusion_full[confusion_full["run_id"].isin(summary["run_id"])]
        if not confusion.empty:
            confusion.to_csv(out_dir / "confusion_matrix_per_class.csv", index=False)
            print(f"{prefix} {len(confusion)} (experiment x class) rows -> confusion_matrix_per_class.csv")

    classifiers = summary["classifier_label"].dropna().unique().tolist()
    default_classifier = classifiers[0] if len(classifiers) == 1 else None

    fig_dir = out_dir / "figures"
    plot_paper_curves(agg, fig_dir, dataset_tag, default_classifier=default_classifier)
    plot_pca_variance(summary, fig_dir, dataset_tag)
    print(f"{prefix} figures -> {fig_dir}")

    # Step (v): suitability per configuration, RAM-budget-gated (step iv's
    # peak memory footprint doubles as the feasibility filter here).
    suitability = compute_suitability(agg, ram_budget_mb=ram_budget_mb,
                                      default_classifier=default_classifier)
    suitability.to_csv(out_dir / "suitability_by_config.csv", index=False)
    suitability_table = build_suitability_table(suitability, dataset_tag)
    suitability_table.to_csv(out_dir / "suitability_table.csv", index=False)
    print(f"{prefix} {len(suitability)} configuration(s) -> suitability_by_config.csv / suitability_table.csv")

    sweet = find_sweet_spot(agg)
    if multi_scenario:
        pca = {}
        for scenario, sub in summary.groupby("scenario_name"):
            for metric, rec in pca_effect(sub).items():
                pca[f"{scenario} / {metric}"] = rec
    else:
        pca = pca_effect(summary)
    write_report(summary, agg, sweet, pca, suitability, dataset_tag, out_dir / "analysis_report.md")
    print(f"{prefix} report -> {out_dir / 'analysis_report.md'}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Analyze KNN benchmark results")
    ap.add_argument("--results-dir", type=Path, default=here.parent / "results")
    ap.add_argument("--out-dir",     type=Path, default=None,
                    help="Default: <results-dir>/analysis")
    ap.add_argument("--ram-budget-mb", type=float, default=512.0,
                    help="On-device RAM budget used as the suitability feasibility "
                         "filter (default: 512 MB, the Pi Zero 2W's physical RAM)")
    ap.add_argument("--no-split-by-dataset", action="store_true",
                    help="Produce one combined figures/report bundle instead of "
                         "one per dataset_name (cifar10 / imagenette / mnist / ...)")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir / "analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] crawling {results_dir}")
    summary = build_summary(results_dir)
    summary.to_csv(out_dir / "experiment_summary.csv", index=False)
    print(f"[analyze] {len(summary)} experiments -> experiment_summary.csv")

    confusion = build_confusion_table(results_dir)
    if not confusion.empty:
        confusion.to_csv(out_dir / "confusion_matrix_per_class.csv", index=False)
        print(f"[analyze] {len(confusion)} (experiment x class) rows -> confusion_matrix_per_class.csv")

    correction_summary = build_correction_summary(summary)
    correction_summary.to_csv(out_dir / "correction_summary.csv", index=False)
    print(f"[analyze] {len(correction_summary)} dataset row(s) -> correction_summary.csv "
         "(clock-offset deviation + baseline correction audit, step i)")

    latency_summary = build_latency_summary(summary)
    latency_summary.to_csv(out_dir / "latency_summary.csv", index=False)
    print(f"[analyze] {len(latency_summary)} dataset row(s) -> latency_summary.csv "
         "(total on-device inference latency, summed over all configs/seeds/sample-sizes)")

    dataset_groups = sorted(summary["dataset_name"].dropna().unique().tolist())

    if not args.no_split_by_dataset and len(dataset_groups) > 1:
        print(f"[analyze] splitting into per-dataset bundles: {dataset_groups}")
        for group in dataset_groups:
            group_summary = summary[summary["dataset_name"] == group].reset_index(drop=True)
            _analyze_group(group_summary, confusion, out_dir / group,
                           args.ram_budget_mb, tag=group)
    else:
        _analyze_group(summary, confusion, out_dir, args.ram_budget_mb)

    print("[analyze] done.")


if __name__ == "__main__":
    main()
