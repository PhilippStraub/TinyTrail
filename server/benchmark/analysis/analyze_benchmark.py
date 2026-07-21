#!/usr/bin/env python3
"""
Benchmark analysis pipeline for the KNN Sweet-Spot project (DBE SS26).

Crawls every results/exp_* folder, extracts one tidy row of metrics per
experiment, aggregates across seeds, and produces the figures + descriptive
analysis the project paper requires.

It works at two levels:

  LEVEL 1  experiment_summary.csv    one row per experiment (the canonical data file)
  LEVEL 2  aggregated_by_config.csv  mean ± std ± 95% CI per (pca, samples_per_class),
                                     averaged over seeds (the replicates)

and writes:

  figures/                 accuracy / energy / memory / cpu / latency vs samples
                           (PCA vs no-PCA, with CI bands) + a 2x2 paper grid
  per_experiment/          per-experiment resource timeline (power+cpu+ram+latency)
  analysis_report.md       descriptive analysis, sweet-spot, PCA effect, methods

Usage (repo venv python):
  python analyze_benchmark.py
  python analyze_benchmark.py --results-dir ../results --out-dir ../results/analysis
  python analyze_benchmark.py --no-per-experiment       # skip per-experiment timelines
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
if "ipykernel" not in sys.modules and not os.environ.get("JPY_PARENT_PID"):
    matplotlib.use("Agg")       # headless CLI; keep notebook backends interactive/inline
import matplotlib.pyplot as plt

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

# Runs excluded from every aggregate: each received a duplicate of the *preceding*
# experiment's result ZIP (byte-identical merged_analysis, same run_id), so the row
# carries the wrong seed's accuracy and latency. Proven on the PC clock alone — the
# whole deploy→result cycle took ~4-5 s while the returned resource log spans a
# 49-114 s inference — so this is a result-attribution failure, not a clock-sync
# artifact. Their energy is already absent (no power.inference_window). Detected by
# analysis/verify_power_coverage.py (check 2, causal containment); per-run evidence in
# results/analysis/power_coverage_audit.csv. Affected cells keep 9 of 10 seeds.
EXCLUDED_EXPERIMENTS = {
    "scenario_cifar10_knn_benchmark/exp_004_pca95var_spc2_s789",
    "scenario_cifar10_knn_benchmark/exp_039_pca95var_spc20_s1618",
    "scenario_imagenette_nearest_centroid/exp_125_nopca_spc2_s1337",
}


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
    window. Falls back to the global config baseline_power_w (or None) if no
    bracketing block is found, e.g. for result sets collected before baseline
    blocks existed.
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
    row["avg_power_W"]              = inf.get("avg_power_W")
    row["net_avg_power_W"]          = inf.get("net_avg_power_W")   # global-baseline, legacy
    row["energy_per_image_J"]       = inf.get("energy_per_image_J")
    row["net_energy_per_image_J"]   = inf.get("net_energy_per_image_J")  # global-baseline, legacy
    row["inference_duration_s"]     = inf.get("duration_s") or full.get("duration_s")

    # ── Local (warm, time-bracketed) baseline correction + EDP ───────────────
    # net_avg_power_W above used the single global config baseline_power_w; this
    # recomputes it from the nearest warm-idle baseline_* blocks in the same
    # scenario dir instead (see _local_baseline_power_w). EDP = P_bar * T_bar,
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

    # ── Clock-offset correction (measured PC<->Pi handshake, see PLAN_ZEIT_HANDSHAKE.md) ──
    # Independent of the local-baseline correction above (idle-power drift vs. clock
    # skew are separate, non-exclusive error sources); combined into *_offset_corrected
    # columns below. clock_offset_s/round_trip_s are always present (even None) so the
    # measured offset can be plotted across the whole campaign as a drift check.
    clock_offset = meta.get("clock_offset") or {}
    row["clock_offset_s"] = clock_offset.get("offset_s")
    row["clock_offset_round_trip_s"] = clock_offset.get("round_trip_s")
    offset_metrics = _offset_corrected_power_metrics(exp_dir, meta, row["n_images"])
    row.update(offset_metrics)
    avg_power_offset = offset_metrics.get("avg_power_W_offset_corrected")
    if avg_power_offset is not None and local_baseline_w is not None:
        row["net_avg_power_W_offset_corrected"] = avg_power_offset - local_baseline_w
    else:
        row["net_avg_power_W_offset_corrected"] = np.nan
    if pd.notna(row["net_avg_power_W_offset_corrected"]) and pd.notna(row["latency_ms_mean"]):
        row["EDP_offset_corrected"] = (
            float(row["net_avg_power_W_offset_corrected"]) * float(row["latency_ms_mean"]) / 1000.0
        )
    else:
        row["EDP_offset_corrected"] = np.nan

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


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 — aggregate across seeds (mean ± std ± 95% CI)
# ──────────────────────────────────────────────────────────────────────────────

METRICS = [
    ("accuracy",              "Accuracy"),
    ("precision_macro",       "Precision (macro)"),
    ("recall_macro",          "Recall (macro)"),
    ("f1_macro",              "F1 (macro)"),
    ("net_energy_per_image_J","Net energy per image (J)"),
    ("avg_power_W",           "Avg power (W)"),
    ("net_avg_power_W",       "Net avg power (W)"),
    ("cpu_percent_mean",      "CPU (%)"),
    ("memory_rss_mb_mean",    "Memory RSS (MB)"),
    ("memory_rss_mb_max",     "Peak memory RSS (MB)"),
    ("classifier_state_kb",   "Classifier state (KB)"),
    ("support_set_kb",        "Reference support budget (KB)"),
    ("latency_ms_mean",       "Latency (ms)"),
    ("latency_sum_ms",        "Total inference time (ms)"),
    ("net_avg_power_W_local", "Net avg power, local baseline (W)"),
    ("EDP",                   "Energy-Delay Product (J)"),
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

def _plot_metric_vs_samples(ax, agg: pd.DataFrame, metric: str, title: str):
    """One metric vs samples-per-class, one line per PCA config, with CI band.

    For EDP, a second hatched band shows the systematic baseline uncertainty
    (EDP_ci95_systematic) alongside the solid seed-CI band — same mean, two
    different, non-additive uncertainty sources. See _local_baseline_sem_w().
    """
    label_col = "plot_label" if "plot_label" in agg.columns else "pca_label"
    has_systematic = metric == "EDP" and "EDP_ci95_systematic" in agg.columns
    for label, sub in agg.groupby(label_col):
        sub = sub.sort_values("samples_per_class")
        x = sub["samples_per_class"].to_numpy()
        y = sub[f"{metric}_mean"].to_numpy()
        ci = sub[f"{metric}_ci95"].to_numpy()
        if np.all(np.isnan(y)):
            continue
        line, = ax.plot(x, y, marker="o", label=label)
        ci_safe = np.nan_to_num(ci, nan=0.0)
        ax.fill_between(x, y - ci_safe, y + ci_safe, alpha=0.18, color=line.get_color())
        if has_systematic:
            ci_sys = np.nan_to_num(sub["EDP_ci95_systematic"].to_numpy(), nan=0.0)
            ax.fill_between(x, y - ci_sys, y + ci_sys, alpha=0.12, hatch="//",
                             facecolor="none", edgecolor=line.get_color())
    ax.set_title(title)
    ax.set_xlabel("Reference samples per class")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    if has_systematic:
        ax.legend(title="Pipeline (band: seed CI; hatched: systematic baseline unc.)")
    else:
        ax.legend(title="Pipeline")


def plot_paper_curves(agg: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Individual figures
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric_vs_samples(ax, agg, metric, title)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{metric}_vs_samples.png", dpi=150)
        plt.close(fig)

    # 2×2 paper grid (accuracy, energy, memory, cpu)
    grid_metrics = [("accuracy", "Accuracy"),
                    ("net_energy_per_image_J", "Net energy / image (J)"),
                    ("memory_rss_mb_mean", "Memory RSS (MB)"),
                    ("cpu_percent_mean", "CPU (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, title) in zip(axes.ravel(), grid_metrics):
        _plot_metric_vs_samples(ax, agg, metric, title)
    fig.suptitle("KNN benchmark — metrics vs reference-sample count (PCA vs no-PCA)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "summary_grid.png", dpi=150)
    plt.close(fig)


def plot_pca_variance(summary: pd.DataFrame, fig_dir: Path):
    sub = summary[summary["pca_enabled"] == True].dropna(subset=["pca_explained_variance_pct"])
    if sub.empty:
        return
    by_comp = sub.groupby("pca_components")["pca_explained_variance_pct"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(int(c)) for c in by_comp.index], by_comp.values)
    for i, v in enumerate(by_comp.values):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Explained variance (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_explained_variance.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Per-experiment resource timeline (the notebook's view, automated for all runs)
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


def per_experiment_timeline(exp_dir: Path, out_dir: Path):
    merged_path = _latest(exp_dir.glob("merged_analysis_*.csv"))
    power_path  = _latest(exp_dir.glob("power_*.csv"))
    if merged_path is None:
        return
    df = pd.read_csv(merged_path)
    if "timestamp_unix" not in df.columns:
        return
    t0 = pd.to_numeric(df["timestamp_unix"], errors="coerce").min()
    df["t_rel"] = pd.to_numeric(df["timestamp_unix"], errors="coerce") - t0

    have_power = power_path is not None
    n_axes = 4 if have_power else 3
    fig, axes = plt.subplots(n_axes, 1, figsize=(13, 2.4 * n_axes), sharex=True)
    ai = 0

    if have_power:
        # Reuse the engine's parser: it converts the naive local ISO timestamps to
        # true epoch seconds via datetime.timestamp() (tz-aware, resolution-safe).
        ts_p, pw_list = _read_power_csv(power_path)
        ts_p = np.asarray(ts_p, dtype=float)
        pw = np.asarray(pw_list, dtype=float)
        t_rel = ts_p - t0
        axes[ai].plot(t_rel, pw, lw=0.8)
        axes[ai].axhline(pw.mean(), ls="--", color="orange", label=f"avg {pw.mean():.2f} W")
        axes[ai].axvline(0, ls=":", color="green", label="inference start")
        axes[ai].set_ylabel("Power (W)"); axes[ai].legend(loc="upper right"); axes[ai].grid(alpha=0.3)
        ai += 1

    cpu = pd.to_numeric(df.get("cpu_percent_total"), errors="coerce")
    axes[ai].plot(df["t_rel"], cpu.rolling(10, min_periods=1).mean(), lw=1.0)
    axes[ai].set_ylabel("CPU (%)"); axes[ai].grid(alpha=0.3); ai += 1

    rss = pd.to_numeric(df.get("process_rss_mb_resource"), errors="coerce")
    axes[ai].plot(df["t_rel"], rss, lw=1.0)
    axes[ai].set_ylabel("RSS (MB)"); axes[ai].grid(alpha=0.3); ai += 1

    lat = pd.to_numeric(df.get("latency_ms"), errors="coerce")
    axes[ai].plot(df["t_rel"], lat, lw=0.5, alpha=0.5)
    axes[ai].plot(df["t_rel"], lat.rolling(20, min_periods=1).mean(), lw=1.5)
    axes[ai].set_ylabel("Latency (ms)"); axes[ai].grid(alpha=0.3)
    axes[ai].set_xlabel("Time since inference start (s)")

    fig.suptitle(exp_dir.name)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{exp_dir.name}_timeline.png", dpi=130)
    plt.close(fig)


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


def find_robust_saturation_point(agg: pd.DataFrame, frac: float = 0.95) -> Dict:
    """Robust saturation point s*: smallest samples-per-class stage whose mean
    accuracy reaches `frac` of the max mean accuracy AND whose margin above that
    threshold exceeds the standard error of the mean (SEM) at that stage.

    The SEM check guards against picking a stage that only clears the threshold
    due to seed noise — the same size-2 cell that happens to score high by luck.
    Uses accuracy_std/n_seeds already produced by aggregate(); a stage without
    that data (n_seeds<2) is accepted on the threshold alone, since robustness
    can't be assessed there.
    """
    out = {}
    label_col = "plot_label" if "plot_label" in agg.columns else "pca_label"
    for label, sub in agg.groupby(label_col):
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
            out[label] = {
                "max_accuracy": max_acc,
                "target_accuracy": float(target),
                "robust_saturation_samples_per_class": s_star,
            }
    return out


def compute_suitability(
    agg: pd.DataFrame,
    s_min: Optional[int] = None,
    ram_budget_mb: Optional[float] = None,
) -> pd.DataFrame:
    """Suitability_c = Acc_bar_c / (EDP_bar_c + sigma_EDP_c) * (s_min / s*_c) per
    configuration c (grouped the same way `agg` already is — typically
    classifier_type x pca_label).

    EDP_bar_c/sigma_EDP_c are the mean/std of the already seed-averaged per-stage
    EDP (aggregate()'s "EDP_mean" column) taken ACROSS samples-per-class stages —
    sigma_EDP_c therefore captures energy-cost growth over the sample-size
    trajectory, not seed-to-seed noise (that's already averaged out per stage).

    Memory is deliberately excluded from the score; instead `feasible` is a
    pass/fail gate against `ram_budget_mb` using the worst case (max over stages)
    of peak RSS per configuration, not the mean — RAM grows monotonically with
    samples_per_class for KNN, so a config can look fine on average and still
    exceed the budget at the largest sample count.
    """
    label_col = "plot_label" if "plot_label" in agg.columns else "pca_label"
    if s_min is None:
        s_min = int(agg["samples_per_class"].min())

    saturation = find_robust_saturation_point(agg, frac=0.95)

    rows = []
    for label, sub in agg.groupby(label_col):
        sub = sub.sort_values("samples_per_class")
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

        mem_col = "memory_rss_mb_max_mean" if "memory_rss_mb_max_mean" in sub.columns else None
        mem_feasibility = sub[mem_col].max() if mem_col else np.nan

        s_star = saturation.get(label, {}).get("robust_saturation_samples_per_class")

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
            "config": label,
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


def pca_effect(summary: pd.DataFrame) -> Dict:
    """Paired comparison PCA vs no-PCA at matched (samples_per_class, seed)."""
    piv = summary.pivot_table(index=["samples_per_class", "seed"],
                              columns="pca_label", values=["accuracy", "net_energy_per_image_J"])
    out = {}
    for metric in ["accuracy", "net_energy_per_image_J"]:
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
                 sweet: Dict, pca: Dict, out_path: Path):
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
    lines.append("- **Energy** is the external Voltcraft measurement, baseline-subtracted "
                 "(`net_energy_per_image_J`) → marginal cost of one inference, independent of duty cycle.")
    lines.append("- **CPU%** is the system-wide, duty-cycled value at the configured rover cadence "
                 "(not peak inference CPU).")
    lines.append("- **Memory (RSS)** includes the Python+TFLite runtime (~tens of MB); the marginal "
                 "store for samples is the analytical `knn_store_kb` column (spc × classes × dim × 4 B).")
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
                 "net_energy_per_image_J_mean", "cpu_percent_mean_mean",
                 "memory_rss_mb_mean_mean", "latency_ms_mean_mean",
                 "EDP_mean", "EDP_ci95", "EDP_ci95_systematic"]
    show_cols = [c for c in show_cols if c in agg.columns]
    lines.append(_table(agg[show_cols].round(4)))
    lines.append("")

    lines.append("## 6. Figures\n")
    lines.append("- `figures/summary_grid.png` — the paper's 2×2 (accuracy/energy/memory/CPU vs samples)")
    lines.append("- `figures/<metric>_vs_samples.png` — individual curves with CI bands")
    lines.append("- `figures/pca_explained_variance.png`")
    lines.append("- `per_experiment/<id>_timeline.png` — power/CPU/RAM/latency over time per run")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Analyze KNN benchmark results")
    ap.add_argument("--results-dir", type=Path, default=here.parent / "results")
    ap.add_argument("--out-dir",     type=Path, default=None,
                    help="Default: <results-dir>/analysis")
    ap.add_argument("--no-per-experiment", action="store_true",
                    help="Skip per-experiment resource timelines")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir / "analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] crawling {results_dir}")
    summary = build_summary(results_dir)
    summary.to_csv(out_dir / "experiment_summary.csv", index=False)
    print(f"[analyze] {len(summary)} experiments -> experiment_summary.csv")

    multi_scenario = summary["scenario_name"].nunique(dropna=True) > 1
    if multi_scenario:
        group_cols = ["scenario_name", "classifier_label", "pca_label",
                      "pca_enabled", "samples_per_class"]
    else:
        group_cols = None
    agg = aggregate(summary, group_cols)
    if multi_scenario:
        agg["plot_label"] = agg["scenario_name"].fillna("unknown") + " / " + agg["pca_label"].fillna("unknown")
    agg.to_csv(out_dir / "aggregated_by_config.csv", index=False)
    print(f"[analyze] {len(agg)} (pca x samples) cells -> aggregated_by_config.csv")

    confusion = build_confusion_table(results_dir)
    if not confusion.empty:
        confusion.to_csv(out_dir / "confusion_matrix_per_class.csv", index=False)
        print(f"[analyze] {len(confusion)} (experiment x class) rows -> confusion_matrix_per_class.csv")

    fig_dir = out_dir / "figures"
    plot_paper_curves(agg, fig_dir)
    plot_pca_variance(summary, fig_dir)
    print(f"[analyze] figures -> {fig_dir}")

    if not args.no_per_experiment:
        pe_dir = out_dir / "per_experiment"
        for exp_dir in _discover_experiment_dirs(results_dir):
            per_experiment_timeline(exp_dir, pe_dir)
        print(f"[analyze] per-experiment timelines -> {pe_dir}")

    sweet = find_sweet_spot(agg)
    if multi_scenario:
        pca = {}
        for scenario, sub in summary.groupby("scenario_name"):
            for metric, rec in pca_effect(sub).items():
                pca[f"{scenario} / {metric}"] = rec
    else:
        pca = pca_effect(summary)
    write_report(summary, agg, sweet, pca, out_dir / "analysis_report.md")
    print(f"[analyze] report -> {out_dir / 'analysis_report.md'}")
    print("[analyze] done.")


if __name__ == "__main__":
    main()
