"""
Post-hoc audit: does every power trace fully cover its inference window?

Walks all experiment folders under a results directory (both layouts:
results/exp_* and results/<scenario>/exp_*) and verifies, per experiment,
that the energy metrics in experiment_metadata.json rest on a sound
alignment between the PC-clock power trace (power_*.csv, Voltcraft
DSP-3005) and the Pi-clock inference window (resource_usage_*.csv).

Checks per experiment
  1. margins      power trace starts before the inference window and ends
                  after it (front/back margin > 0; small margins are WARN).
  2. containment  causal bound on the Pi-PC clock offset without trusting
                  NTP: the first Pi sample happened after the deploy started
                  (PC event started_at) and the last Pi sample before the
                  result ZIP arrived (PC event result_received_at), so
                  t1 - result_received_at <= offset <= t0 - started_at.
                  An empty interval, or one that excludes 0, proves a clock
                  inconsistency -> FAIL.
  3. gaps         no sampling holes: consecutive resource rows ~0.2 s apart
                  (detects an NTP step on the Pi mid-run; also checks
                  monotonicity), and no holes in the power trace inside the
                  window (detects a dying PowerLogger).
  4. run_id       exactly one resource_usage CSV, and its run_id matches the
                  merged_analysis file from the same run.
  5. xcorr        clock-free offset estimate: cross-correlate the Pi's
                  cpu_percent_total with the PC's P_out; the peak lag is the
                  residual Pi-PC offset (n/a when the signals are too flat).
  6. sensitivity  shift the integration window by +/- shift seconds and
                  report the worst relative change of the window energy;
                  also recompute the stored energy_J as a self-check.
  7. handshake    sanity-check the actively measured PC<->Pi clock offset
                  (experiment_metadata.json / baseline_summary.json's
                  "clock_offset.offset_s", from the NTP-style round-trip
                  handshake in benchmark_pipeline.py::_measure_clock_offset) ->
                  WARN when |offset_s| exceeds --offset-warn. This is the
                  "measured vs. sanity bound" check described in
                  MESSMETHODIK_ZEITSYNCHRONISATION.md §5 point 7.

Usage (from server/benchmark/, with the venv python):
    python analysis/verify_power_coverage.py                 # audit ./results
    python analysis/verify_power_coverage.py path/to/results
    python analysis/verify_power_coverage.py --csv-out audit.csv

Writes a per-experiment CSV (default <results>/analysis/power_coverage_audit.csv)
and prints a summary with the aggregate numbers to quote in the paper.
Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = no experiments found.

Note: power/metadata timestamps are naive PC-local ISO strings; run this on a
machine configured with the same timezone as the benchmark PC so that
datetime.fromisoformat().timestamp() reproduces the original epoch values.
Pure stdlib, no third-party dependencies.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"

# ──────────────────────────────────────────────────────────────────────────────
# Loading (same parsing rules as benchmark_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def read_power_csv(power_csv: Path) -> Tuple[List[float], List[float]]:
    """Parallel lists (epoch_seconds, power_W); unparseable rows skipped."""
    ts, pw = [], []
    with open(power_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.fromisoformat(row["timestamp"]).timestamp()
                p = float(row["P_out"])
            except (ValueError, KeyError):
                continue
            ts.append(t)
            pw.append(p)
    return ts, pw


def read_resource_csv(res_csv: Path) -> Tuple[List[float], List[float], set]:
    """(timestamp_unix, cpu_percent_total, run_ids) in file order."""
    ts, cpu, run_ids = [], [], set()
    with open(res_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["timestamp_unix"])
            except (ValueError, KeyError, TypeError):
                continue
            ts.append(t)
            try:
                cpu.append(float(row.get("cpu_percent_total", "")))
            except (ValueError, TypeError):
                cpu.append(0.0)
            if row.get("run_id"):
                run_ids.add(row["run_id"])
    return ts, cpu, run_ids


def read_metadata(exp_dir: Path) -> Dict:
    meta_path = exp_dir / "experiment_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _iso_to_epoch(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _run_id_from_name(path: Path, prefix: str) -> str:
    """resource_usage_20250101_120000.csv -> 20250101_120000"""
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


# ──────────────────────────────────────────────────────────────────────────────
# Numerics (trapezoid identical to benchmark_pipeline._integrate)
# ──────────────────────────────────────────────────────────────────────────────

def integrate_energy(ts: List[float], pw: List[float],
                     lo: Optional[float] = None, hi: Optional[float] = None) -> Optional[float]:
    sel = [(t, p) for t, p in zip(ts, pw)
           if (lo is None or t >= lo) and (hi is None or t <= hi)]
    if len(sel) < 2:
        return None
    return sum((pa + pb) / 2.0 * (tb - ta)
               for (ta, pa), (tb, pb) in zip(sel, sel[1:]))


def _resample(xs: List[float], ys: List[float], grid: List[float]) -> List[Optional[float]]:
    """Linear interpolation of (xs, ys) onto grid; None outside the data range."""
    out: List[Optional[float]] = []
    j = 0
    n = len(xs)
    for g in grid:
        if g < xs[0] or g > xs[-1]:
            out.append(None)
            continue
        while j < n - 2 and xs[j + 1] < g:
            j += 1
        x0, x1 = xs[j], xs[j + 1]
        if x1 <= x0:
            out.append(ys[j])
            continue
        w = (g - x0) / (x1 - x0)
        out.append(ys[j] * (1.0 - w) + ys[j + 1] * w)
    return out


def _pearson(a: List[Optional[float]], b: List[Optional[float]],
             min_pairs: int = 50) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < min_pairs:
        return None
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    return sxy / math.sqrt(sxx * syy)


def xcorr_offset(p_ts: List[float], p_w: List[float],
                 r_ts: List[float], r_cpu: List[float],
                 step: float = 0.2, max_lag: float = 30.0
                 ) -> Tuple[Optional[float], Optional[float]]:
    """Estimate the Pi-PC clock offset by cross-correlating CPU load with power.

    Positive result = the Pi clock runs ahead of the PC clock. Returns
    (offset_seconds, peak_correlation) or (None, None) when the signals do
    not overlap enough or carry no usable structure (near-constant load).
    """
    if len(p_ts) < 10 or len(r_ts) < 10:
        return None, None
    lo = max(p_ts[0], r_ts[0] - max_lag)
    hi = min(p_ts[-1], r_ts[-1] + max_lag)
    if hi - lo < 20 * step:
        return None, None
    n = int((hi - lo) / step) + 1
    grid = [lo + i * step for i in range(n)]
    P = _resample(p_ts, p_w, grid)
    C = _resample(r_ts, r_cpu, grid)

    k_max = int(max_lag / step)
    best_corr, best_k = None, None
    for k in range(-k_max, k_max + 1):
        # pairs (P[i], C[i+k]) -> the lag k*step at peak correlation is the offset
        if k >= 0:
            corr = _pearson(P[:n - k], C[k:])
        else:
            corr = _pearson(P[-k:], C[:n + k])
        if corr is not None and (best_corr is None or corr > best_corr):
            best_corr, best_k = corr, k
    if best_corr is None or best_corr < 0.3:
        return None, best_corr
    return best_k * step, best_corr


# ──────────────────────────────────────────────────────────────────────────────
# Per-experiment audit
# ──────────────────────────────────────────────────────────────────────────────

def audit_experiment(exp_dir: Path, args) -> Dict:
    rec: Dict = {
        "experiment": str(exp_dir.relative_to(args.results)),
        "status": "PASS",
        "reasons": [],
    }
    fails: List[str] = []
    warns: List[str] = []

    power_csv = next(iter(sorted(exp_dir.glob("power_*.csv"))), None)
    if power_csv is None:
        rec["status"] = "SKIP"
        rec["reasons"] = ["no power_*.csv (power disabled or run not kept)"]
        return rec

    p_ts, p_w = read_power_csv(power_csv)
    if len(p_ts) < 2:
        rec["status"] = "FAIL"
        rec["reasons"] = ["power CSV has fewer than 2 parseable rows"]
        return rec

    res_csvs = sorted(exp_dir.glob("resource_usage_*.csv"))
    merged = sorted(exp_dir.glob("merged_analysis_*.csv"))
    merged_run_id = _run_id_from_name(merged[0], "merged_analysis_") if merged else None

    # Check 4: exactly one resource CSV, run_id consistent with merged_analysis
    if not res_csvs:
        rec["status"] = "FAIL"
        rec["reasons"] = ["no resource_usage_*.csv -> window unverifiable "
                          "(pipeline note: check clock sync)"]
        return rec
    res_csv = res_csvs[0]
    if len(res_csvs) > 1:
        matching = [p for p in res_csvs
                    if _run_id_from_name(p, "resource_usage_") == merged_run_id]
        if matching:
            res_csv = matching[0]
            warns.append(f"{len(res_csvs)} resource CSVs; using run_id match "
                         f"{res_csv.name}")
        else:
            fails.append(f"{len(res_csvs)} resource CSVs, none matches "
                         f"merged_analysis run_id {merged_run_id}")
    r_ts, r_cpu, run_ids = read_resource_csv(res_csv)
    res_run_id = _run_id_from_name(res_csv, "resource_usage_")
    if merged_run_id and res_run_id != merged_run_id:
        fails.append(f"run_id mismatch: resource={res_run_id} "
                     f"merged={merged_run_id}")
    if run_ids and len(run_ids) > 1:
        fails.append(f"resource CSV mixes run_ids: {sorted(run_ids)}")

    if len(r_ts) < 2:
        rec["status"] = "FAIL"
        rec["reasons"] = fails + ["resource CSV has fewer than 2 timestamp_unix rows"]
        return rec

    t0, t1 = min(r_ts), max(r_ts)
    rec["window_s"] = round(t1 - t0, 3)

    # Check 1: coverage margins (power trace must bracket the window)
    front = t0 - p_ts[0]
    back = p_ts[-1] - t1
    rec["margin_front_s"] = round(front, 3)
    rec["margin_back_s"] = round(back, 3)
    if front <= 0:
        fails.append(f"power trace starts {-front:.2f}s AFTER window start")
    elif front < args.min_margin:
        warns.append(f"front margin only {front:.2f}s")
    if back <= 0:
        fails.append(f"power trace ends {-back:.2f}s BEFORE window end")
    elif back < args.min_margin:
        warns.append(f"back margin only {back:.2f}s")

    # Check 2: causal containment -> provable bound on the clock offset
    meta = read_metadata(exp_dir)
    started = _iso_to_epoch(meta.get("started_at"))
    received = _iso_to_epoch(meta.get("result_received_at"))
    if started is not None and received is not None:
        off_hi = t0 - started      # offset cannot exceed this
        off_lo = t1 - received     # offset cannot be below this
        rec["offset_bound_lo_s"] = round(off_lo, 3)
        rec["offset_bound_hi_s"] = round(off_hi, 3)
        if off_lo > off_hi:
            fails.append(f"causally impossible timestamps "
                         f"(offset bound [{off_lo:.1f}, {off_hi:.1f}] empty)")
        elif off_lo > 0 or off_hi < 0:
            dist = max(off_lo, -off_hi)
            fails.append(f"clocks provably offset by >= {dist:.1f}s "
                         f"(bound [{off_lo:.1f}, {off_hi:.1f}] excludes 0)")
    else:
        warns.append("no started_at/result_received_at in metadata "
                     "-> containment check skipped")

    # Check 7: sanity-check the actively measured handshake offset (see
    # MESSMETHODIK_ZEITSYNCHRONISATION.md §5 point 7). Independent of Check 2:
    # this reads the NTP-style round-trip measurement itself rather than
    # deriving a bound from started_at/result_received_at.
    clock_offset = meta.get("clock_offset") or {}
    measured_offset = clock_offset.get("offset_s")
    rec["clock_offset_s"] = measured_offset
    rec["clock_offset_round_trip_s"] = clock_offset.get("round_trip_s")
    if measured_offset is None:
        warns.append("no clock_offset in metadata -> handshake check skipped")
    elif abs(measured_offset) > args.offset_warn:
        warns.append(f"measured clock offset {measured_offset:+.3f}s exceeds "
                     f"{args.offset_warn:.1f}s sanity bound")

    # Check 3a: Pi sampling gaps / monotonicity (detects NTP step mid-run)
    diffs = [b - a for a, b in zip(r_ts, r_ts[1:])]
    neg = sum(1 for d in diffs if d < 0)
    max_gap_pi = max(diffs) if diffs else 0.0
    rec["pi_max_gap_s"] = round(max_gap_pi, 3)
    if neg:
        fails.append(f"resource timestamps non-monotonic ({neg} negative steps "
                     f"-> clock stepped backwards)")
    if max_gap_pi > args.max_gap:
        fails.append(f"resource sampling gap {max_gap_pi:.1f}s "
                     f"(expected ~{1.0 / args.rate:.1f}s -> clock step or stall)")

    # Check 3b: power trace holes inside the window (detects dying PowerLogger)
    in_win = [t for t in p_ts if t0 - 0.5 <= t <= t1 + 0.5]
    gaps_w = [b - a for a, b in zip(in_win, in_win[1:])]
    max_gap_pw = max(gaps_w) if gaps_w else float("inf")
    rec["power_max_gap_s"] = round(max_gap_pw, 3) if gaps_w else None
    expected = (t1 - t0) * args.rate + 1
    coverage = len(in_win) / expected if expected > 0 else 0.0
    rec["power_coverage"] = round(coverage, 3)
    if not gaps_w:
        fails.append("no power samples inside the inference window")
    elif max_gap_pw > args.max_gap:
        fails.append(f"power trace hole of {max_gap_pw:.1f}s inside window")
    elif coverage < 0.9:
        warns.append(f"only {coverage:.0%} of expected power samples in window")

    # Check 5: clock-free offset estimate via CPU-load / power cross-correlation
    off, peak = xcorr_offset(p_ts, p_w, r_ts, r_cpu,
                             step=1.0 / args.rate, max_lag=args.max_lag)
    rec["xcorr_offset_s"] = round(off, 2) if off is not None else None
    rec["xcorr_peak"] = round(peak, 3) if peak is not None else None
    if off is not None and abs(off) > args.offset_warn:
        warns.append(f"cross-correlation offset {off:+.1f}s (peak r={peak:.2f})")

    # Check 6: boundary sensitivity + self-check against stored energy_J
    e0 = integrate_energy(p_ts, p_w, t0, t1)
    if e0 and e0 > 0:
        deltas = []
        for s in (args.shift, -args.shift):
            es = integrate_energy(p_ts, p_w, t0 + s, t1 + s)
            if es is not None:
                deltas.append(abs(es - e0) / e0 * 100.0)
        if deltas:
            rec["sensitivity_pct"] = round(max(deltas), 3)
            if max(deltas) > args.sensitivity_warn:
                warns.append(f"window shift +/-{args.shift:.0f}s changes energy "
                             f"by {max(deltas):.2f}%")
        stored = (meta.get("power", {}) or {}).get("inference_window", {}) or {}
        stored_e = stored.get("energy_J")
        if stored_e:
            drift = abs(e0 - stored_e) / stored_e * 100.0
            if drift > 0.5:
                warns.append(f"recomputed energy differs from metadata by "
                             f"{drift:.1f}% (stale files?)")

    if fails:
        rec["status"] = "FAIL"
    elif warns:
        rec["status"] = "WARN"
    rec["reasons"] = fails + warns
    return rec


# ──────────────────────────────────────────────────────────────────────────────
# Discovery / reporting
# ──────────────────────────────────────────────────────────────────────────────

def discover_experiments(results_dir: Path) -> List[Path]:
    """Every folder holding a power trace or experiment metadata, any depth."""
    dirs = {p.parent for p in results_dir.rglob("power_*.csv")}
    dirs |= {p.parent for p in results_dir.rglob("experiment_metadata.json")}
    return sorted(dirs)


CSV_FIELDS = [
    "experiment", "status", "window_s", "margin_front_s", "margin_back_s",
    "offset_bound_lo_s", "offset_bound_hi_s", "clock_offset_s",
    "clock_offset_round_trip_s", "pi_max_gap_s",
    "power_max_gap_s", "power_coverage", "xcorr_offset_s", "xcorr_peak",
    "sensitivity_pct", "reasons",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit power-trace coverage of the inference windows")
    ap.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS_DIR,
                    help=f"results directory (default: {DEFAULT_RESULTS_DIR})")
    ap.add_argument("--rate", type=float, default=8.0,
                    help="sampling rate of power and resource logs in Hz (default 5)")
    ap.add_argument("--max-gap", type=float, default=1.0,
                    help="FAIL on sampling gaps above this many seconds (default 1.0)")
    ap.add_argument("--min-margin", type=float, default=1.0,
                    help="WARN when trace margins are below this (default 1.0 s)")
    ap.add_argument("--shift", type=float, default=2.0,
                    help="window shift for the sensitivity check (default 2.0 s)")
    ap.add_argument("--sensitivity-warn", type=float, default=1.0,
                    help="WARN when the shifted energy differs by more than this %% (default 1.0)")
    ap.add_argument("--max-lag", type=float, default=30.0,
                    help="cross-correlation search range in seconds (default 30)")
    ap.add_argument("--offset-warn", type=float, default=1.0,
                    help="WARN when |xcorr offset| or |measured handshake clock_offset| "
                         "exceeds this (default 1.0 s, see MESSMETHODIK_ZEITSYNCHRONISATION.md "
                         "§3 for why 1s is the point the effect on energy becomes noticeable)")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="output CSV (default <results>/analysis/power_coverage_audit.csv)")
    args = ap.parse_args()

    args.results = args.results.resolve()
    if not args.results.is_dir():
        print(f"Results directory not found: {args.results}")
        return 2
    exp_dirs = discover_experiments(args.results)
    if not exp_dirs:
        print(f"No experiment folders under {args.results}")
        return 2

    records = [audit_experiment(d, args) for d in exp_dirs]

    for r in records:
        line = f"[{r['status']:<4}] {r['experiment']}"
        extras = []
        if r.get("margin_front_s") is not None:
            extras.append(f"margins {r['margin_front_s']:+.1f}/"
                          f"{r['margin_back_s']:+.1f}s")
        if r.get("clock_offset_s") is not None:
            extras.append(f"handshake offset {r['clock_offset_s']:+.3f}s")
        if r.get("xcorr_offset_s") is not None:
            extras.append(f"xcorr offset {r['xcorr_offset_s']:+.1f}s")
        if r.get("sensitivity_pct") is not None:
            extras.append(f"sens {r['sensitivity_pct']:.2f}%")
        if extras:
            line += "  (" + ", ".join(extras) + ")"
        print(line)
        for reason in r["reasons"]:
            print(f"         - {reason}")

    audited = [r for r in records if r["status"] != "SKIP"]
    n_fail = sum(1 for r in records if r["status"] == "FAIL")
    n_warn = sum(1 for r in records if r["status"] == "WARN")
    n_skip = sum(1 for r in records if r["status"] == "SKIP")
    print(f"\n{'=' * 60}")
    print(f"AUDIT: {len(audited)} experiments checked  "
          f"({len(audited) - n_fail - n_warn} PASS, {n_warn} WARN, "
          f"{n_fail} FAIL, {n_skip} skipped)")

    # Aggregate numbers to quote in the paper
    def _vals(key):
        return [r[key] for r in audited if r.get(key) is not None]
    if audited:
        margins_f, margins_b = _vals("margin_front_s"), _vals("margin_back_s")
        offsets = [abs(v) for v in _vals("xcorr_offset_s")]
        measured_offsets = [abs(v) for v in _vals("clock_offset_s")]
        sens = _vals("sensitivity_pct")
        gaps = _vals("pi_max_gap_s")
        print("\nPaper numbers (across all audited experiments):")
        if margins_f and margins_b:
            print(f"  trace margins        : front >= {min(margins_f):.1f}s, "
                  f"back >= {min(margins_b):.1f}s")
        if gaps:
            print(f"  max Pi sampling gap  : {max(gaps):.2f}s "
                  f"(nominal {1.0 / args.rate:.1f}s)")
        if measured_offsets:
            n_over = sum(1 for v in measured_offsets if v > args.offset_warn)
            print(f"  measured clock offset: median {sorted(measured_offsets)[len(measured_offsets) // 2]:.3f}s, "
                  f"max {max(measured_offsets):.3f}s  (NTP-style handshake, "
                  f"{len(measured_offsets)}/{len(audited)} runs measured, "
                  f"{n_over} above {args.offset_warn:.1f}s sanity bound)")
        if offsets:
            print(f"  residual clock offset: median {sorted(offsets)[len(offsets) // 2]:.2f}s, "
                  f"max {max(offsets):.2f}s  (cross-correlation, "
                  f"{len(offsets)}/{len(audited)} runs with usable signal)")
        if sens:
            print(f"  window sensitivity   : energy changes <= {max(sens):.2f}% "
                  f"under +/-{args.shift:.0f}s shift")

    csv_out = args.csv_out or (args.results / "analysis" / "power_coverage_audit.csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            row["reasons"] = "; ".join(r["reasons"])
            w.writerow(row)
    print(f"\nPer-experiment table -> {csv_out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
