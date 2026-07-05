#!/usr/bin/env python3
"""Boxplot of inference power (Watt) vs reference sample size.

Project requirement: show the distribution of power usage over sample size as
boxplots instead of a single line with a CI band. Each box is the spread across
random seeds for one plotted series and samples_per_class cell.

The helper is importable from notebooks and supports both layouts:

  results/exp_*
  results/<scenario_name>/exp_*

By default it keeps all complete cells and only de-duplicates repeated
(scenario, pipeline, samples_per_class, seed) records. Use --latest-date-only
when a directory intentionally contains repeated benchmark sessions and only the
newest run date should be plotted.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
if "ipykernel" not in sys.modules and not os.environ.get("JPY_PARENT_PID"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METRIC_LABELS = {
    "net_avg_power_W": "Net power during inference (W) [avg - idle baseline]",
    "avg_power_W": "Power draw during inference (W) [total]",
}

CLASSIFIER_LABELS = {
    "LightweightKNN": "KNN",
    "NearestCentroidClassifier": "Nearest Centroid",
    "KMeansPrototypeClassifier": "KMeans",
}

PREFERRED_PCA_ORDER = ["nopca", "pca20", "pca31", "pca40", "pca80", "pca95var"]
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#d62728",
]

# Keep classifier pairs visually related: no-PCA = darker, PCA = lighter.
CLASSIFIER_PCA_COLORS = {
    ("KNN", "nopca"): "#174A7E",
    ("KNN", "pca"): "#6BAED6",
    ("Nearest Centroid", "nopca"): "#1B7837",
    ("Nearest Centroid", "pca"): "#74C476",
    ("KMeans", "nopca"): "#A63603",
    ("KMeans", "pca"): "#FDB863",
}


def classifier_label(classifier_type: Optional[str]) -> str:
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


def _discover_experiment_dirs(results_dir: Path) -> List[Path]:
    candidates = list(results_dir.glob("exp_*")) + list(results_dir.glob("*/exp_*"))
    return sorted({p.resolve() for p in candidates if p.is_dir()})


def _record_date(meta: dict, exp_dir: Path) -> Optional[str]:
    started = meta.get("started_at")
    if started:
        return str(started)[:10].replace("-", "")
    rc = sorted(exp_dir.glob("resource_usage_*.csv"))
    m = re.search(r"(\d{8})", rc[0].name) if rc else None
    return m.group(1) if m else None


def load_records(results_dir: str | Path, metric: str) -> List[dict]:
    """Load one power record per completed experiment with the requested metric."""
    recs = []
    for exp_dir in _discover_experiment_dirs(Path(results_dir)):
        meta_p = exp_dir / "experiment_metadata.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        inf = (meta.get("power") or {}).get("inference_window") or {}
        val = inf.get(metric)
        if val is None:
            continue

        scenario = meta.get("scenario_name")
        if not scenario and exp_dir.parent.name.startswith("scenario_"):
            scenario = exp_dir.parent.name
        dataset = meta.get("dataset_name") or _infer_dataset_name(scenario)
        pipeline = meta.get("pca_label", "pca" if meta.get("pca_enabled") else "nopca")
        ctype = meta.get("classifier_type")
        clabel = classifier_label(ctype)
        exp_id = meta.get("experiment_id", exp_dir.name)
        recs.append({
            "run_id": f"{scenario}/{exp_id}" if scenario else exp_id,
            "experiment_id": exp_id,
            "scenario": scenario,
            "dataset": dataset,
            "classifier_type": ctype,
            "classifier_label": clabel,
            "pipeline": pipeline,
            "series": f"{clabel} / {pipeline}",
            "spc": meta.get("samples_per_class"),
            "seed": meta.get("seed"),
            "date": _record_date(meta, exp_dir),
            "val": float(val),
        })
    return recs


def deduplicate_records(recs: Sequence[dict]) -> Tuple[List[dict], dict]:
    """Keep the newest duplicate for each scenario/pipeline/spc/seed cell."""
    best = {}
    provenance = defaultdict(set)
    for r in recs:
        key = (r.get("scenario"), r.get("pipeline"), r.get("spc"), r.get("seed"))
        old = best.get(key)
        if old is None or (r.get("date") or "") >= (old.get("date") or ""):
            best[key] = r
    out = sorted(best.values(), key=lambda r: (
        str(r.get("dataset")), str(r.get("scenario")), str(r.get("pipeline")),
        int(r.get("spc") or -1), int(r.get("seed") or -1),
    ))
    for r in out:
        provenance[(r.get("scenario"), r.get("pipeline"))].add(r.get("date"))
    return out, {k: sorted(v) for k, v in provenance.items()}


def keep_latest_dedup(recs: Sequence[dict]) -> Tuple[List[dict], dict]:
    """Compatibility mode: latest run date per scenario/pipeline, then de-duplicate."""
    latest = {}
    for r in recs:
        key = (r.get("scenario"), r.get("pipeline"))
        latest[key] = max(latest.get(key, "0"), r.get("date") or "0")

    filtered = [r for r in recs
                if (r.get("date") or "0") == latest[(r.get("scenario"), r.get("pipeline"))]]
    out, _ = deduplicate_records(filtered)
    return out, latest


def _series_order(series: Iterable[str]) -> List[str]:
    def key(label: str):
        classifier = label.split(" / ")[0]
        pca = label.split(" / ")[-1]
        pca_i = PREFERRED_PCA_ORDER.index(pca) if pca in PREFERRED_PCA_ORDER else 99
        return classifier, pca_i, pca
    return sorted(set(series), key=key)


def color_for_series(label: str) -> str:
    """Stable hue per classifier, darker for no-PCA and lighter for PCA variants."""
    classifier, _, pca = label.partition(" / ")
    pca_group = "nopca" if pca == "nopca" else "pca"
    return CLASSIFIER_PCA_COLORS.get((classifier, pca_group), "#7f7f7f")


def plot_power_boxplot(
    recs: Sequence[dict],
    out: str | Path,
    metric: str = "net_avg_power_W",
    title: Optional[str] = None,
    series_key: Callable[[dict], str] | str = "series",
    series_colors: Optional[dict] = None,
    figsize: Tuple[float, float] = (15, 6),
):
    """Write a grouped power boxplot and return (fig, ax, out_path)."""
    if not recs:
        raise ValueError("No records to plot")

    if isinstance(series_key, str):
        key_name = series_key
        get_series = lambda r: str(r.get(key_name))
    else:
        get_series = series_key

    series_labels = _series_order(get_series(r) for r in recs)
    spcs = sorted({r["spc"] for r in recs if r.get("spc") is not None})
    data = defaultdict(lambda: defaultdict(list))
    for r in recs:
        data[get_series(r)][r["spc"]].append(r["val"])

    fig, ax = plt.subplots(figsize=figsize)
    n_series = len(series_labels)
    group_w = 0.84
    box_w = group_w / max(n_series, 1)
    color_map = {}
    for i, s in enumerate(series_labels):
        if series_colors and s in series_colors:
            color_map[s] = series_colors[s]
        else:
            color_map[s] = color_for_series(s) or PALETTE[i % len(PALETTE)]

    for i, label in enumerate(series_labels):
        positions, values = [], []
        for j, spc in enumerate(spcs):
            vals = data[label][spc]
            if vals:
                positions.append(j + (i - (n_series - 1) / 2) * box_w)
                values.append(vals)
        if not values:
            continue
        color = color_map[label]
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=box_w * 0.85,
            patch_artist=True,
            manage_ticks=False,
            showmeans=True,
            medianprops=dict(color="black"),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=4),
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.62, edgecolor=color)

    ax.set_xticks(range(len(spcs)))
    ax.set_xticklabels(spcs)
    ax.set_xlabel("Reference samples per class")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or "Power usage vs sample size - distribution across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    handles = [mpatches.Patch(facecolor=color_map[s], alpha=0.62, label=s)
               for s in series_labels]
    ax.legend(handles=handles, title="Classifier / PCA", ncol=2, fontsize=8)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig, ax, out_path


def filter_records(recs: Sequence[dict], dataset: Optional[str] = None,
                   scenarios: Optional[Sequence[str]] = None) -> List[dict]:
    scenarios_set = set(scenarios or [])
    out = []
    for r in recs:
        if dataset and r.get("dataset") != dataset:
            continue
        if scenarios_set and r.get("scenario") not in scenarios_set:
            continue
        out.append(r)
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Power boxplot vs sample size")
    ap.add_argument("--results-dir", default=str(here.parent / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="net_avg_power_W", choices=list(METRIC_LABELS))
    ap.add_argument("--dataset", default=None,
                    help="Optional dataset filter, e.g. cifar10 or imagenette")
    ap.add_argument("--latest-date-only", action="store_true",
                    help="Keep only the latest date per scenario/pipeline")
    args = ap.parse_args()

    recs = load_records(args.results_dir, args.metric)
    recs = filter_records(recs, dataset=args.dataset)
    if not recs:
        raise SystemExit(f"No power records with '{args.metric}' under {args.results_dir}")

    if args.latest_date_only:
        recs, provenance = keep_latest_dedup(recs)
        mode = "latest date only"
    else:
        recs, provenance = deduplicate_records(recs)
        mode = "deduplicated cells"

    print(f"Provenance ({mode}):")
    for (scenario, pipeline), dates in sorted(provenance.items(), key=lambda x: str(x[0])):
        n = sum(1 for r in recs if r.get("scenario") == scenario and r.get("pipeline") == pipeline)
        print(f"  {scenario or '-':35} {pipeline:8} dates={dates} n={n}")

    out = args.out or Path(args.results_dir) / "analysis" / "figures" / f"power_boxplot_{args.metric}.png"
    _, _, out_path = plot_power_boxplot(recs, out, metric=args.metric)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
