#!/usr/bin/env python3
"""Regenerate the figures embedded in FINDINGS.md from the benchmark results.

Reuses analyze_benchmark.py for data extraction so the numbers match the notebook /
headless report. Writes PNGs into ./figures/.

Usage:
  python make_findings_figures.py
  python make_findings_figures.py --results-dir ../results
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_benchmark as ab

COLORS = {"nopca": "tab:blue", "pca31": "tab:orange"}


def _color(label):
    return COLORS.get(label)


def main():
    ap = argparse.ArgumentParser(description="Regenerate FINDINGS.md figures")
    ap.add_argument("--results-dir", type=Path, default=HERE.parent / "results")
    ap.add_argument("--fig-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = ab.build_summary(args.results_dir.resolve())
    agg = ab.aggregate(summary)
    N = len(ab.CLASS_LABELS)

    # fig1 — accuracy vs samples
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, sub in agg.groupby("pca_label"):
        sub = sub.sort_values("samples_per_class")
        x, y = sub.samples_per_class.to_numpy(), sub.accuracy_mean.to_numpy()
        ci = np.nan_to_num(sub.accuracy_ci95.to_numpy(), nan=0.0)
        ln, = ax.plot(x, y, marker="o", label=lbl, color=_color(lbl))
        ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=ln.get_color())
    ax.axvspan(10, 20, color="grey", alpha=0.10)
    ax.annotate("practical sweet spot\n(~95% of max)", xy=(15, 0.70), ha="center",
                fontsize=9, color="dimgray")
    ax.set_xlabel("Reference samples per class"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs reference-sample count (CIFAR-10, KNN)")
    ax.grid(alpha=0.3); ax.legend(title="Pipeline")
    fig.savefig(fig_dir / "fig1_accuracy_vs_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig2 — KNN store size (no-PCA uses the raw extractor dim)
    raw_dim = None
    for cand in HERE.parents[2].glob("server/**/scenario_cifar10_knn_benchmark/model/feature_processor*.npz"):
        try:
            raw_dim = int(np.load(cand, allow_pickle=True)["input_dim"]); break
        except Exception:
            continue
    raw_dim = raw_dim or 160
    dims = {}
    for lbl, sub in summary.groupby("pca_label"):
        pc = sub.pca_components.dropna()
        dims[lbl] = int(pc.iloc[0]) if len(pc) else raw_dim
    spc = np.array(sorted(summary.samples_per_class.unique()))
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl in sorted(dims):
        ax.plot(spc, [s * N * dims[lbl] * 4 / 1024 for s in spc], marker="o",
                label=f"{lbl} (dim={dims[lbl]})", color=_color(lbl))
    ax.set_xlabel("Reference samples per class"); ax.set_ylabel("KNN store size (KB)")
    ax.set_title("Reference-set memory cost  (deterministic: spc x 10 x dim x 4 B)")
    ax.grid(alpha=0.3); ax.legend(title="Pipeline")
    fig.savefig(fig_dir / "fig2_knn_store_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig3 — PCA explained variance
    pr = summary[summary.pca_enabled == True].dropna(subset=["pca_explained_variance_pct"])
    if not pr.empty:
        ev = pr.groupby("pca_components").pca_explained_variance_pct.mean()
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.bar([f"PCA({int(c)})" for c in ev.index], ev.values, color="tab:green", width=0.5)
        for b, v in zip(bars, ev.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom")
        ax.set_ylim(0, 105); ax.set_ylabel("Explained variance (%)")
        ax.set_title("PCA explained variance (of 160-D features)"); ax.grid(alpha=0.3, axis="y")
        fig.savefig(fig_dir / "fig3_pca_explained_variance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # fig4 — charge per run (gross + net), @ 5 V
    j_per_mah = 5.0 * 3.6
    summary["charge_run_mAh"] = summary.energy_per_image_J * summary.n_images / j_per_mah
    summary["net_charge_run_mAh"] = summary.net_energy_per_image_J * summary.n_images / j_per_mah

    def _cagg(col):
        rows = []
        for (lbl, s), g in summary.groupby(["pca_label", "samples_per_class"]):
            v = g[col].to_numpy(float)
            rows.append({"pca_label": lbl, "samples_per_class": s,
                         "mean": np.nanmean(v), "ci95": ab._ci95(v)})
        return pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for axx, (col, t) in zip(axes, [("charge_run_mAh", "Gross charge / run"),
                                    ("net_charge_run_mAh", "Net charge / run (marginal)")]):
        a = _cagg(col)
        for lbl, sub in a.groupby("pca_label"):
            sub = sub.sort_values("samples_per_class")
            x, y = sub.samples_per_class.to_numpy(), sub["mean"].to_numpy()
            ci = np.nan_to_num(sub.ci95.to_numpy(), nan=0.0)
            ln, = axx.plot(x, y, marker="o", label=lbl, color=_color(lbl))
            axx.fill_between(x, y - ci, y + ci, alpha=0.18, color=ln.get_color())
        axx.set_xlabel("Reference samples per class"); axx.set_ylabel("Charge per run (mAh)")
        axx.set_title(f"{t}  (@ 5 V)"); axx.grid(alpha=0.3); axx.legend(title="Pipeline")
    fig.suptitle("Charge consumed per run vs sample size", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "fig4_charge_per_run.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig5 — paper 2x2 grid (reuse engine)
    ab.plot_paper_curves(agg, fig_dir)

    print(f"[findings] wrote figures -> {fig_dir}")
    for p in sorted(fig_dir.glob("fig*.png")) + [fig_dir / "summary_grid.png"]:
        if p.exists():
            print("  ", p.name)


if __name__ == "__main__":
    main()
