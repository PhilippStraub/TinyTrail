#!/usr/bin/env python3
"""Boxplot of inference power (Watt) vs reference sample size, per pipeline.

Project requirement: show the *distribution* of energy usage (in watts) over sample
size as boxplots, rather than a single line + CI band. Each box is the spread across
the random seeds for one (pipeline, samples_per_class) cell.

Data hygiene: `results/` can hold runs from several sessions. To avoid mixing
pre-/post-fix runs (and double-counting repeated cells), this script keeps only the
LATEST run date per pipeline and de-duplicates each (pipeline, spc, seed). It prints
the provenance it used so the figure is auditable.

Usage (repo venv python):
  python plot_power_boxplot.py                      # net power, default results dir
  python plot_power_boxplot.py --metric avg_power_W # total power draw instead
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METRIC_LABELS = {
    "net_avg_power_W": "Net power per inference (W)   [avg − idle baseline]",
    "avg_power_W":     "Power draw during inference (W)   [total]",
}
# Fixed pipeline → colour so the legend is stable regardless of which are present.
PIPELINE_ORDER = ["nopca", "pca20", "pca40", "pca80"]
PIPELINE_COLOR = {"nopca": "#1f77b4", "pca20": "#9467bd",
                  "pca40": "#ff7f0e", "pca80": "#2ca02c"}


def load_records(results_dir: str, metric: str):
    recs = []
    for d in sorted(glob.glob(os.path.join(results_dir, "exp_*"))):
        meta_p = os.path.join(d, "experiment_metadata.json")
        if not os.path.isdir(d) or not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p))
        rc = glob.glob(os.path.join(d, "resource_usage_*.csv"))
        m = re.search(r"(\d{8})", os.path.basename(rc[0])) if rc else None
        date = m.group(1) if m else None
        inf = (meta.get("power") or {}).get("inference_window") or {}
        val = inf.get(metric)
        if val is None:
            continue
        recs.append(dict(pipeline=meta.get("pca_label"),
                         spc=meta.get("samples_per_class"),
                         seed=meta.get("seed"), date=date, val=float(val)))
    return recs


def keep_latest_dedup(recs):
    """Keep only each pipeline's latest run date, de-duplicate (pipeline,spc,seed)."""
    latest = {}
    for r in recs:
        latest[r["pipeline"]] = max(latest.get(r["pipeline"], "0"), r["date"] or "0")
    seen, out = set(), []
    for r in recs:
        if r["date"] != latest[r["pipeline"]]:
            continue
        key = (r["pipeline"], r["spc"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out, latest


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Power boxplot vs sample size")
    ap.add_argument("--results-dir", default=str(here.parent / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="net_avg_power_W", choices=list(METRIC_LABELS))
    args = ap.parse_args()

    recs = load_records(args.results_dir, args.metric)
    if not recs:
        raise SystemExit(f"No power records with '{args.metric}' under {args.results_dir}")
    recs, latest = keep_latest_dedup(recs)

    pipes = [p for p in PIPELINE_ORDER if any(r["pipeline"] == p for r in recs)]
    pipes += sorted({r["pipeline"] for r in recs} - set(pipes), key=str)
    spcs = sorted({r["spc"] for r in recs})

    data = defaultdict(lambda: defaultdict(list))
    for r in recs:
        data[r["pipeline"]][r["spc"]].append(r["val"])

    print("Provenance (latest run date used per pipeline):")
    for p in pipes:
        ns = {s: len(data[p][s]) for s in spcs if data[p][s]}
        print(f"  {p:7} date={latest[p]}  n-per-spc={ns}")

    # Grouped boxplots: x = sample size, one coloured box per pipeline within each group.
    fig, ax = plt.subplots(figsize=(13, 6))
    n_p = len(pipes)
    group_w = 0.8
    box_w = group_w / n_p
    for i, p in enumerate(pipes):
        positions, series = [], []
        for j, s in enumerate(spcs):
            vals = data[p][s]
            if vals:
                positions.append(j + (i - (n_p - 1) / 2) * box_w)
                series.append(vals)
        if not series:
            continue
        c = PIPELINE_COLOR.get(p, plt.cm.tab10(i % 10))
        bp = ax.boxplot(series, positions=positions, widths=box_w * 0.85,
                        patch_artist=True, manage_ticks=False, showmeans=True,
                        medianprops=dict(color="black"),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=4))
        for box in bp["boxes"]:
            box.set(facecolor=c, alpha=0.6, edgecolor=c)

    ax.set_xticks(range(len(spcs)))
    ax.set_xticklabels(spcs)
    ax.set_xlabel("Reference samples per class")
    ax.set_ylabel(METRIC_LABELS[args.metric])
    ax.set_title("Energy usage (Watt) vs sample size — distribution across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    handles = [mpatches.Patch(facecolor=PIPELINE_COLOR.get(p, "gray"), alpha=0.6,
                              label=f"{p} ({latest[p]})") for p in pipes]
    ax.legend(handles=handles, title="Pipeline (run date)")

    out = args.out or os.path.join(args.results_dir, "analysis", "figures",
                                   f"power_boxplot_{args.metric}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
