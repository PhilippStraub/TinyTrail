"""Does the measured power actually change at the inference-window boundaries?

Complements analysis/verify_power_coverage.py. That script bounds the Pi-PC clock
offset causally (which is airtight but loose, because the bound is padded by the
deploy and transfer durations) and additionally tries to estimate the offset by
cross-correlating CPU load with power — an estimator that fails on this data,
because CPU load is saturated and therefore featureless during inference.

This script uses the one feature that is not featureless: the power *edge*. The
Pi loads the TFLite model and the KNN state before inference (a CPU-heavy burst
that draws clearly more than inference itself), and goes idle after the result
has been merged and transmitted. Both transitions are sharp. If the two clocks
agreed, those transitions sit exactly on the window boundaries t0 and t1 taken
from the Pi's resource log; a clock offset of delta would displace them by delta.

A single run is too noisy to locate an edge. Stacking every run of a dataset
family on its own t0 (superposed-epoch averaging) removes the noise: the
ensemble mean over several hundred runs resolves the edge to the sampling
period. The reported edge position is therefore a direct, clock-free measurement
of the residual offset.

Method
  1. Bin every power sample by its offset to t0 (and to t1) at --bin resolution.
  2. Average each bin over all runs of the dataset family.
  3. Locate the t0 edge as the falling edge of the setup burst: the crossing of
     the midpoint between the burst peak and the inference plateau.
  4. Report the crossing position relative to the window boundary.

Usage (from server/benchmark/, with the venv python):
    python analysis/verify_window_alignment.py
    python analysis/verify_window_alignment.py --bin 0.2 --csv-out align.csv

Note: like verify_power_coverage.py, power timestamps are naive PC-local ISO
strings, so run this in the benchmark PC's timezone. Pure stdlib.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"

# Same misattributed runs excluded by analysis/analyze_benchmark.py.
EXCLUDED = {
    "exp_004_pca95var_spc2_s789",
    "exp_039_pca95var_spc20_s1618",
    "exp_125_nopca_spc2_s1337",
}

FAMILIES = [("scenario_cifar10_*", "CIFAR-10"),
            ("scenario_imagenette_*", "Imagenette")]


def read_power(path: str) -> Tuple[List[float], List[float]]:
    ts, pw = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(datetime.fromisoformat(row["timestamp"]).timestamp())
                pw.append(float(row["P_out"]))
            except (ValueError, KeyError):
                continue
    return ts, pw


def read_window(exp_dir: str) -> Optional[Tuple[float, float]]:
    g = sorted(glob.glob(os.path.join(exp_dir, "resource_usage_*.csv")))
    if not g:
        return None
    unix = []
    with open(g[0], newline="") as f:
        for row in csv.DictReader(f):
            try:
                unix.append(float(row["timestamp_unix"]))
            except (ValueError, KeyError, TypeError):
                continue
    return (min(unix), max(unix)) if unix else None


def stack(results: Path, pattern: str, span: float, binw: float
          ) -> Tuple[Dict[float, float], Dict[float, float], int]:
    """Ensemble-mean power binned by offset to t0 and to t1."""
    at_t0, at_t1 = defaultdict(list), defaultdict(list)
    n = 0
    for exp_dir in sorted(glob.glob(str(results / pattern / "exp_*"))):
        if os.path.basename(exp_dir) in EXCLUDED:
            continue
        pg = sorted(glob.glob(os.path.join(exp_dir, "power_*.csv")))
        if not pg:
            continue
        win = read_window(exp_dir)
        if not win:
            continue
        t0, t1 = win
        ts, pw = read_power(pg[0])
        if len(ts) < 50:
            continue
        n += 1
        for t, p in zip(ts, pw):
            d0 = t - t0
            if -span <= d0 <= span:
                at_t0[round(d0 / binw) * binw].append(p)
            d1 = t - t1
            if -span <= d1 <= span:
                at_t1[round(d1 / binw) * binw].append(p)
    mean0 = {k: statistics.mean(v) for k, v in at_t0.items() if len(v) > 20}
    mean1 = {k: statistics.mean(v) for k, v in at_t1.items() if len(v) > 20}
    return mean0, mean1, n


def locate_falling_edge(curve: Dict[float, float], plateau_lo: float,
                        plateau_hi: float) -> Optional[Tuple[float, float, float]]:
    """Find where the setup burst falls to the inference plateau.

    Returns (crossing_time, peak_power, plateau_power). The crossing is where the
    curve drops through the midpoint between the burst peak and the plateau,
    linearly interpolated between the two bracketing samples.
    """
    keys = sorted(curve)
    plateau_pts = [curve[k] for k in keys if plateau_lo <= k <= plateau_hi]
    if not plateau_pts:
        return None
    plateau = statistics.mean(plateau_pts)
    pre = [(k, curve[k]) for k in keys if k < 0]
    if not pre:
        return None
    peak_t, peak = max(pre, key=lambda kv: kv[1])
    if peak <= plateau:
        return None
    mid = (peak + plateau) / 2.0
    fwd = [k for k in keys if k >= peak_t]
    for a, b in zip(fwd, fwd[1:]):
        if curve[a] >= mid > curve[b]:
            span = curve[a] - curve[b]
            frac = (curve[a] - mid) / span if span > 0 else 0.0
            return a + frac * (b - a), peak, plateau
    return None


def sparkline(curve: Dict[float, float], lo: float, hi: float, width: int = 44
              ) -> List[str]:
    keys = [k for k in sorted(curve) if lo <= k <= hi]
    if not keys:
        return []
    vals = [curve[k] for k in keys]
    vmin, vmax = min(vals), max(vals)
    out = []
    for k in keys:
        v = curve[k]
        bars = int((v - vmin) / (vmax - vmin) * width) if vmax > vmin else 0
        mark = "  <-- window boundary" if abs(k) < 1e-9 else ""
        out.append(f"    {k:+05.1f}s  {v:.4f} W  {'#' * bars}{mark}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify that power transitions coincide with window edges")
    ap.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--bin", type=float, default=0.2,
                    help="bin width in seconds (default 0.2 = sampling period)")
    ap.add_argument("--span", type=float, default=10.0,
                    help="seconds analysed either side of a boundary (default 10)")
    ap.add_argument("--plot-span", type=float, default=4.0,
                    help="seconds shown in the printed profile (default 4)")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="write the ensemble profiles to this CSV")
    args = ap.parse_args()

    args.results = args.results.resolve()
    if not args.results.is_dir():
        print(f"Results directory not found: {args.results}")
        return 2

    rows = []
    verdicts = []
    for pattern, label in FAMILIES:
        mean0, mean1, n = stack(args.results, pattern, args.span, args.bin)
        if not mean0:
            print(f"[{label}] no runs found — skipped\n")
            continue
        print(f"=== {label}  ({n} runs stacked, {args.bin:.1f}s bins) ===")
        print("  ensemble mean power around t0 (window start):")
        for line in sparkline(mean0, -args.plot_span, args.plot_span):
            print(line)

        edge = locate_falling_edge(mean0, 2.0, args.span)
        if edge:
            crossing, peak, plateau = edge
            verdicts.append((label, crossing))
            print(f"\n  setup burst peaks at {peak:.3f} W, inference plateau "
                  f"{plateau:.3f} W")
            print(f"  --> falling edge crosses the midpoint at t0{crossing:+.2f}s")
            print(f"      residual Pi-PC clock offset |delta| <= {abs(crossing):.2f}s "
                  f"(sampling period {args.bin:.1f}s)")
        else:
            print("\n  no usable setup burst — edge not locatable")
        print()

        for k in sorted(mean0):
            rows.append({"dataset": label, "aligned_on": "t0",
                         "offset_s": round(k, 3), "mean_power_W": round(mean0[k], 5)})
        for k in sorted(mean1):
            rows.append({"dataset": label, "aligned_on": "t1",
                         "offset_s": round(k, 3), "mean_power_W": round(mean1[k], 5)})

    if verdicts:
        worst = max(abs(c) for _, c in verdicts)
        print("=" * 62)
        print("Paper number: across dataset families, the power transition "
              f"coincides with\nthe inference-window start to within {worst:.2f}s "
              "(ensemble of all runs).")
        print("A clock offset of several seconds would displace it by that amount.")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "aligned_on",
                                              "offset_s", "mean_power_W"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nEnsemble profiles -> {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
