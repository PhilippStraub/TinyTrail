#!/usr/bin/env python3
"""
Automated Benchmark Pipeline for KNN Sweet Spot Analysis.

Sweeps the cross-product of  (PCA config) × (samples-per-class) × (seed).
For each experiment this pipeline:
  1. Trains the feature processor (cached — only once per PCA config)
  2. Creates the KNN initial state with the given sample count + seed
  3. Patches config.json and copies artifacts to the deployment packager
  4. Subscribes to NATS analysis.logs.{device_id} to receive results directly
  5. Starts external Voltcraft DSP-3005 power logging (if enabled)
  6. Deploys the model to the device via NATS JetStream
  7. Waits for the merged_analysis CSV to arrive in the log ZIP
  8. Computes energy/power metrics and saves CSV + metadata to benchmark/results/

Prerequisites:
  - Backbone model already trained (model_object.tflite + labels_object.txt present)
  - NATS server running and reachable from both this machine and the Pi
  - deployment-receiver.py running on the Pi (Analytical-Data-Processor NOT required)
  - (optional) Voltcraft DSP-3005 on a COM port for energy measurement

Usage:
  python benchmark_pipeline.py                        # run sweep with benchmark_config.json
  python benchmark_pipeline.py --config my_cfg.json
  python benchmark_pipeline.py --dry-run              # print experiments, no execution
  python benchmark_pipeline.py --measure-baseline 60  # measure idle Pi power, then exit
"""

import argparse
import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile as zf_module
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Project paths  (all relative to this file: server/benchmark/)
# ──────────────────────────────────────────────────────────────────────────────
BENCHMARK_DIR   = Path(__file__).parent
SERVER_DIR      = BENCHMARK_DIR.parent
PROJECT_ROOT    = SERVER_DIR.parent

RETRAINER_ROOT      = PROJECT_ROOT / "server/Retrainer/on-device-drift-detection-main"
PY_EXAMPLE_DIR      = RETRAINER_ROOT / "examples/python_examples/scenario_cifar10_knn_benchmark"
MODEL_TRAINING_DIR  = RETRAINER_ROOT / "examples/model_training/scenario_cifar10_knn_benchmark"
TEST_DATA_DIR       = RETRAINER_ROOT / "examples/dataset_cifar10/test"   # held-out CIFAR-10 test split

PACKAGER_MAIN   = PROJECT_ROOT / "server/Deployment-Packager/nats-model-service/src/nats_model_service/main.py"
RAW_DATA_DIR    = PROJECT_ROOT / "server/Deployment-Packager/nats-model-service/raw_data/scenario_cifar10_knn_benchmark"

DEFAULT_CONFIG      = BENCHMARK_DIR / "benchmark_config.json"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_subprocess(
    cmd: List[Any],
    cwd: Optional[Path] = None,
    label: str = "",
    env: Optional[Dict[str, str]] = None,
) -> None:
    tag = label or Path(str(cmd[1])).stem
    print(f"\n[{tag}] $ {' '.join(str(c) for c in cmd)}")
    # Force UTF-8 in children so unicode prints (e.g. "→", "…") don't crash on a
    # Windows cp1252 console — these scripts are shared and pepper logs with them.
    child_env = {**(env or os.environ), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=child_env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed (exit {result.returncode}): {tag}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_benchmark_config(config_path: Path) -> Dict:
    with open(config_path) as f:
        return json.load(f)


def build_experiment_list(cfg: Dict) -> List[Dict]:
    """Expand benchmark_config into a flat ordered list of experiments.

    Order: pca_config → samples_per_class → seed
    This groups all seeds for one (pca, spc) pair together so variance
    across seeds is immediately visible in the results directory.
    """
    experiments = []
    seeds = cfg.get("seeds", [42])
    for pca_cfg in cfg["pca_configs"]:
        pca_enabled    = pca_cfg.get("enabled", True)
        pca_components = pca_cfg.get("components") if pca_enabled else None
        pca_label      = f"pca{pca_components}" if pca_enabled else "nopca"
        for spc in cfg["samples_per_class"]:
            for seed in seeds:
                experiments.append({
                    "pca_enabled":       pca_enabled,
                    "pca_components":    pca_components,
                    "pca_label":         pca_label,
                    "samples_per_class": spc,
                    "seed":              seed,
                })
    return experiments


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 – Feature processor training
# ──────────────────────────────────────────────────────────────────────────────

def ensure_feature_processor(model_dir: Path, pca_components: int) -> Path:
    """Return path to the cached feature_processor_pca{N}.npz, training it if needed."""
    cached = model_dir / f"feature_processor_pca{pca_components}.npz"
    if cached.exists():
        print(f"[FP] feature_processor_pca{pca_components}.npz already cached — skipping.")
        return cached

    print(f"\n[FP] Training feature processor (PCA components = {pca_components}) ...")
    make_fp = MODEL_TRAINING_DIR / "make_feature_processor.py"
    run_subprocess(
        [sys.executable, make_fp, "--pca-components", str(pca_components), "--output-dir", model_dir],
        label=f"make_feature_processor(pca={pca_components})",
    )
    default_fp = model_dir / "feature_processor.npz"
    if not default_fp.exists():
        raise FileNotFoundError(f"Expected {default_fp} after training — not found.")
    shutil.copy2(default_fp, cached)
    print(f"[FP] Cached as {cached.name}")
    return cached


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 – Initial state creation
# ──────────────────────────────────────────────────────────────────────────────

_cis_module = None

def _get_cis_module():
    """Import the scenario's create_initial_state once and cache it.

    The first call pays the TensorFlow import (~9 s) a single time for the whole
    sweep; every later experiment reuses the module — and its cached TFLite
    interpreter — in-process, with no per-experiment subprocess or re-import.
    """
    global _cis_module
    if _cis_module is None:
        if str(PY_EXAMPLE_DIR) not in sys.path:
            sys.path.insert(0, str(PY_EXAMPLE_DIR))
        # In-process prints from create_initial_state may contain non-cp1252 chars;
        # make stdout/stderr UTF-8 so a Windows console can't crash on them (the
        # subprocess path used to get this via PYTHONUTF8).
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
        import importlib
        _cis_module = importlib.import_module("create_initial_state")
    return _cis_module


def create_initial_state(
    samples_per_class: int,
    pca_enabled: bool,
    feature_processor_path: Optional[Path],
    seed: int = 42,
) -> None:
    # In-process call (was a subprocess) so the whole sweep pays the TF import once.
    cis = _get_cis_module()
    print(f"\n[create_initial_state(spc={samples_per_class},pca={pca_enabled},seed={seed})] (in-process)")
    cis.build_initial_state(
        samples_per_class=samples_per_class,
        seed=seed,
        no_pca=not pca_enabled,
        feature_processor_path=(str(feature_processor_path)
                                if (pca_enabled and feature_processor_path) else None),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 – Deployment preparation
# ──────────────────────────────────────────────────────────────────────────────

def _load_base_config() -> Dict:
    template = BENCHMARK_DIR / "config_template.json"
    if not template.exists():
        src = RAW_DATA_DIR / "config.json"
        shutil.copy2(src, template)
        print(f"[DEPLOY] Saved config_template.json from {src.relative_to(PROJECT_ROOT)}")
    with open(template) as f:
        return json.load(f)


def populate_test_images(seed: int, per_class: int) -> int:
    """Rebuild raw_data/input_images from the held-out TEST split, seeded.

    Evaluates on novelty (never-seen) data, and — because the draw depends on the
    seed — different seeds cover different test images, expanding effective test
    coverage across the sweep. Returns the total number of images placed.
    """
    import random

    dst_dir = RAW_DATA_DIR / "input_images"
    if dst_dir.exists():
        shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not TEST_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Test split not found: {TEST_DATA_DIR}\n"
            f"Run  python export_cifar10.py  first to create the train/test split."
        )

    total = 0
    for ci, class_dir in enumerate(sorted(p for p in TEST_DATA_DIR.iterdir() if p.is_dir())):
        files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        if not files:
            continue
        # Per-class deterministic draw: same (seed, class) -> same images; the
        # +ci decorrelates classes so they don't all pick the same indices.
        rng = random.Random(seed + ci)
        rng.shuffle(files)
        chosen = files[:per_class]
        out_class = dst_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)
        for f in chosen:
            shutil.copy2(f, out_class / f.name)
        total += len(chosen)
    return total


def prepare_deployment(
    model_dir: Path,
    pca_enabled: bool,
    pca_components: Optional[int],
    samples_per_class: int,
    experiment_id: str,
    seed: int = 42,
    test_images_per_class: int = 0,
) -> None:
    """Copy all artifacts to raw_data and write a patched config.json."""
    dst_model = RAW_DATA_DIR / "model"
    dst_state = RAW_DATA_DIR / "initial_state"

    # Rebuild the test set from the held-out split (if configured)
    if test_images_per_class and test_images_per_class > 0:
        n = populate_test_images(seed, test_images_per_class)
        print(f"[DEPLOY] input_images: {n} test images "
              f"({test_images_per_class}/class, seed={seed}) from TEST split")

    if pca_enabled and pca_components:
        fp_src = model_dir / f"feature_processor_pca{pca_components}.npz"
        shutil.copy2(fp_src, dst_model / "feature_processor.npz")

    src_state = PY_EXAMPLE_DIR / "initial_state"
    shutil.copy2(src_state / "knn_initial_state_objects.json",
                 dst_state / "knn_initial_state_objects.json")
    shutil.copy2(src_state / "knn_reference_stats.json",
                 dst_state / "knn_reference_stats.json")

    config = _load_base_config()
    config["tinylcm"]["feature_transformation"]["enabled"] = pca_enabled

    known_classes = config["tinylcm"]["drift_detectors"][0].get("stable_known_classes", [])
    num_classes = len(known_classes) if known_classes else 10
    config["tinylcm"]["adaptive_classifier"]["max_samples"] = samples_per_class * num_classes

    # Same seed for test-image shuffle so each run is reproducible end-to-end
    config["image_input"]["shuffle_seed"] = seed

    with open(RAW_DATA_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(RAW_DATA_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "experiment_id":         experiment_id,
            "pca_enabled":           pca_enabled,
            "pca_components":        pca_components,
            "samples_per_class":     samples_per_class,
            "seed":                  seed,
            "test_images_per_class": test_images_per_class,
            "deployed_at":           datetime.now().isoformat(),
        }, f, indent=2)

    print(f"[DEPLOY] Artifacts ready for {experiment_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 – Deploy + collect result via NATS (no Analytical-Data-Processor needed)
# ──────────────────────────────────────────────────────────────────────────────

async def _subscribe_deploy_collect(
    nats_url: str,
    device_id: str,
    experiment_id: str,
    timeout_seconds: int,
    results_dir: Path,
    power_cfg: Optional[Dict] = None,
) -> Optional[Path]:
    """
    1. Subscribe to analysis.logs.{device_id} (deliver_policy=new)
    2. Deploy in a thread executor so the event loop stays alive
    3. Wait for a ZIP containing merged_analysis_{experiment_id}*.csv
    4. Save all files from the ZIP into results_dir/{experiment_id}/
    """
    from nats.aio.client import Client as NATS
    from nats.js.api import ConsumerConfig, AckPolicy, DeliverPolicy, StreamConfig

    nc = NATS()
    await nc.connect(nats_url)
    js = nc.jetstream()

    # ANALYSIS stream must use `limits` retention so we can subscribe with
    # DeliverPolicy.NEW.  WorkQueue retention forbids it for push consumers.
    # If the stream exists with workqueue retention (created by transmitter),
    # delete and recreate it. Messages already in the queue are lost, but they
    # belong to old experiments and aren't needed.
    stream_cfg = StreamConfig(
        name="ANALYSIS",
        subjects=["analysis.logs.*"],
        storage="file",
        retention="limits",
        max_msg_size=8 * 1024 * 1024,
    )
    try:
        info = await js.stream_info("ANALYSIS")
        retention_str = str(info.config.retention).lower()
        if "workqueue" in retention_str or "work_queue" in retention_str:
            print("[RESULT] ANALYSIS stream is workqueue — recreating with limits retention...")
            await js.delete_stream("ANALYSIS")
            await js.add_stream(stream_cfg)
            print("[RESULT] ANALYSIS stream recreated.")
    except Exception:
        print("[RESULT] ANALYSIS stream not found — creating with limits retention.")
        await js.add_stream(stream_cfg)

    subject = f"analysis.logs.{device_id}"
    # Short unique consumer name (NATS has a 256-char limit)
    consumer_name = f"bench-{int(time.time()) % 10000000}"

    received_csv: Optional[Path] = None
    done_event = asyncio.Event()

    async def handle_msg(msg):
        nonlocal received_csv
        try:
            with zf_module.ZipFile(io.BytesIO(msg.data)) as zf:
                names = zf.namelist()
                # DeliverPolicy.NEW guarantees this ZIP arrived after our deployment,
                # so any merged_analysis CSV in it belongs to the current experiment.
                match = next(
                    (n for n in names if "merged_analysis" in n),
                    None,
                )
                if match:
                    exp_dir = ensure_dir(results_dir / experiment_id)
                    csv_path = exp_dir / Path(match).name
                    csv_path.write_bytes(zf.read(match))
                    print(f"[RESULT] CSV received: {csv_path.name}")
                    # Save all other log files from the same ZIP
                    for name in names:
                        if name != match:
                            try:
                                (exp_dir / Path(name).name).write_bytes(zf.read(name))
                            except Exception:
                                pass
                    received_csv = csv_path
                    done_event.set()
            await msg.ack()
        except Exception as e:
            print(f"[RESULT] Error processing message: {e}")
            try:
                await msg.nak()
            except Exception:
                pass

    config = ConsumerConfig(
        ack_policy=AckPolicy.EXPLICIT,
        deliver_policy=DeliverPolicy.NEW,
    )
    sub = await js.subscribe(subject, durable=consumer_name, config=config, cb=handle_msg)
    print(f"[RESULT] Subscribed. Deploying experiment {experiment_id} ...")

    # Start fresh: wipe any artifacts from a previous run of this experiment id, so
    # the folder only ever holds one run's files. Otherwise timestamped logs from
    # earlier runs accumulate and the analysis can't tell which run is current.
    exp_dir = results_dir / experiment_id
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
    ensure_dir(exp_dir)

    # Start external power logging (out-of-band, on this PC — zero load on the Pi).
    # Begins before deploy so we capture the full window; sliced to the inference
    # sub-window later via the Pi's resource_usage timestamps.
    power_logger = None
    if power_cfg and power_cfg.get("enabled"):
        try:
            from dsp3005_reader import PowerLogger
            power_csv = exp_dir / f"power_{experiment_id}.csv"
            power_logger = PowerLogger(
                power_csv,
                port=power_cfg.get("port", "COM3"),
                rate=power_cfg.get("rate", 5.0),
            )
            power_logger.start()
            print(f"[POWER] Logging → {power_csv.name}  (device: {power_logger.device})")
        except Exception as e:
            print(f"[POWER] Could not start power logger: {e} — continuing without power.")
            power_logger = None

    # Deploy in executor — subprocess.run() is blocking, but runs in a thread
    # so the asyncio event loop continues processing incoming NATS messages
    deploy_env = {**os.environ, "NATS_URL": nats_url}
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [str(sys.executable), str(PACKAGER_MAIN), "scenario_cifar10_knn_benchmark"],
            env=deploy_env,
        ),
    )
    print(f"[RESULT] Deployed. Waiting for result (timeout={timeout_seconds}s) ...")

    try:
        await asyncio.wait_for(done_event.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"[RESULT] Timeout — no result received for {experiment_id}")

    # Stop power logging
    if power_logger is not None:
        power_logger.stop()
        msg = f"[POWER] Stopped — {power_logger.n_rows} samples."
        if power_logger.error:
            msg += f"  (error: {power_logger.error})"
        print(msg)

    # Clean up consumer
    try:
        await sub.unsubscribe()
    except Exception:
        pass
    try:
        await js.delete_consumer("ANALYSIS", consumer_name)
    except Exception:
        pass
    await nc.drain()

    return received_csv


def collect_result(
    nats_url: str,
    device_id: str,
    experiment_id: str,
    timeout_seconds: int,
    results_dir: Path,
    power_cfg: Optional[Dict] = None,
) -> Optional[Path]:
    """Synchronous wrapper — runs the async subscribe+deploy+collect in a fresh event loop."""
    return asyncio.run(_subscribe_deploy_collect(
        nats_url, device_id, experiment_id, timeout_seconds, results_dir, power_cfg,
    ))


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 – Energy / power metrics from the external Voltcraft DSP-3005 log
# ──────────────────────────────────────────────────────────────────────────────

def _read_power_csv(power_csv: Path):
    """Return parallel lists (epoch_seconds, power_W) from a dsp3005 log."""
    import csv as _csv
    ts, pw = [], []
    with open(power_csv, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                # DSP timestamps are PC-local ISO; .timestamp() → epoch (same PC clock)
                t = datetime.fromisoformat(row["timestamp"]).timestamp()
                p = float(row["P_out"])
            except (ValueError, KeyError):
                continue
            ts.append(t)
            pw.append(p)
    return ts, pw


def _inference_window(exp_dir: Path):
    """Absolute [t0, t1] epoch window of actual inference, from the Pi resource log."""
    res_csv = next(iter(exp_dir.glob("resource_usage_*.csv")), None)
    if res_csv is None:
        return None
    import csv as _csv
    unix = []
    with open(res_csv, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                unix.append(float(row["timestamp_unix"]))
            except (ValueError, KeyError):
                continue
    return (min(unix), max(unix)) if unix else None


def _integrate(ts, pw, lo=None, hi=None):
    """Trapezoidal energy (J) + mean power (W) over an optional [lo, hi] window."""
    sel = [(t, p) for t, p in zip(ts, pw)
           if (lo is None or t >= lo) and (hi is None or t <= hi)]
    if len(sel) < 2:
        return None
    energy = sum((pa + pb) / 2.0 * (tb - ta)
                 for (ta, pa), (tb, pb) in zip(sel, sel[1:]))
    duration = sel[-1][0] - sel[0][0]
    return {
        "energy_J":    round(energy, 4),
        "avg_power_W": round(energy / duration, 4) if duration > 0 else None,
        "duration_s":  round(duration, 3),
        "samples":     len(sel),
    }


def compute_power_metrics(
    exp_dir: Path,
    n_images: Optional[int] = None,
    baseline_power_w: Optional[float] = None,
) -> Dict:
    """Derive energy/power metrics for one experiment from its power_*.csv.

    Produces:
      - full_window:       integral over the whole deploy→result trace
      - inference_window:  integral sliced to the Pi's actual inference period
      - energy_per_image_J and net_* (marginal over idle baseline), when available
    """
    power_csv = next(iter(exp_dir.glob("power_*.csv")), None)
    if power_csv is None:
        return {}
    ts, pw = _read_power_csv(power_csv)
    if len(ts) < 2:
        return {}

    metrics: Dict = {}
    full = _integrate(ts, pw)
    if full:
        metrics["full_window"] = full

    window = _inference_window(exp_dir)
    inf = _integrate(ts, pw, *window) if window else None
    if inf:
        if n_images:
            inf["energy_per_image_J"] = round(inf["energy_J"] / n_images, 6)
        if baseline_power_w is not None and inf["avg_power_W"] is not None:
            net_w = inf["avg_power_W"] - baseline_power_w
            inf["net_avg_power_W"] = round(net_w, 4)
            inf["net_energy_J"] = round(net_w * inf["duration_s"], 4)
            if n_images:
                inf["net_energy_per_image_J"] = round(net_w * inf["duration_s"] / n_images, 6)
        metrics["inference_window"] = inf
    elif not window:
        metrics["note"] = "no resource_usage_*.csv → inference window unavailable; check clock sync"

    return metrics


def _count_inferences(result_csv: Path) -> Optional[int]:
    """Number of inference rows in the merged_analysis CSV (= images processed)."""
    try:
        import csv as _csv
        with open(result_csv, newline="") as f:
            return sum(1 for _ in _csv.DictReader(f))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def clean_experiment_results(results_dir: Path) -> int:
    """Delete every per-experiment folder (``exp_*``) from ``results_dir``.

    Leaves ``analysis/`` and any other files (baseline, summary) untouched — only
    the experiment runs are removed. Use before a fresh sweep so orphaned folders
    from a previous config (whose indices / sample sizes have since shifted) don't
    linger and get mixed into the analysis, which crawls every ``exp_*`` folder.
    """
    exp_dirs = sorted(p for p in results_dir.glob("exp_*") if p.is_dir())
    for p in exp_dirs:
        shutil.rmtree(p, ignore_errors=True)
    print(f"[CLEAN] Removed {len(exp_dirs)} previous experiment folder(s); "
          f"kept analysis/ and other files.")
    return len(exp_dirs)


def run_pipeline(cfg: Dict, results_dir: Path, dry_run: bool = False, clean: bool = False) -> None:
    device_id      = cfg["device_id"]
    nats_url       = cfg.get("nats_url", "nats://localhost:4222")
    result_timeout = cfg.get("result_timeout_seconds", 1800)
    model_dir      = PY_EXAMPLE_DIR / "model"
    experiments    = build_experiment_list(cfg)
    total          = len(experiments)

    power_cfg      = cfg.get("power", {})
    baseline_w     = power_cfg.get("baseline_power_w")
    test_per_class = cfg.get("test_images_per_class", 0)

    print(f"\n{'='*60}")
    print(f"KNN BENCHMARK PIPELINE  —  {total} experiments")
    print(f"Device : {device_id}")
    print(f"NATS   : {nats_url}")
    print(f"Results: {results_dir}")
    if test_per_class:
        print(f"Test   : {test_per_class} images/class from held-out test split (seeded per run)")
    if power_cfg.get("enabled"):
        print(f"Power  : {power_cfg.get('port', 'COM3')} @ {power_cfg.get('rate', 5.0)} Hz"
              + (f"  (idle baseline {baseline_w} W)" if baseline_w is not None else ""))
    print(f"{'='*60}")

    if dry_run:
        for i, exp in enumerate(experiments):
            print(f"  [{i+1:03d}] {exp['pca_label']}  spc={exp['samples_per_class']}  seed={exp['seed']}")
        print("\n(dry-run: no experiments executed)")
        return

    if clean:
        clean_experiment_results(results_dir)

    # Pre-train all required feature processors before the sweep begins
    needed_pca = {
        exp["pca_components"]
        for exp in experiments
        if exp["pca_enabled"] and exp["pca_components"] is not None
    }
    for pca_n in sorted(needed_pca):
        ensure_feature_processor(model_dir, pca_n)

    summary: List[Dict] = []

    for i, exp in enumerate(experiments):
        pca_enabled       = exp["pca_enabled"]
        pca_components    = exp["pca_components"]
        samples_per_class = exp["samples_per_class"]
        pca_label         = exp["pca_label"]
        seed              = exp["seed"]
        idx               = f"{i+1:03d}"
        experiment_id     = f"exp_{idx}_{pca_label}_spc{samples_per_class}_s{seed}"

        print(f"\n{'─'*60}")
        print(f"[{idx}/{total}]  {experiment_id}")
        print(f"  PCA  : enabled={pca_enabled}, components={pca_components}")
        print(f"  Samples/class: {samples_per_class}  Seed: {seed}")
        print(f"{'─'*60}")

        fp_path = (model_dir / f"feature_processor_pca{pca_components}.npz"
                   if pca_enabled and pca_components else None)

        # Step 1: Create initial state
        create_initial_state(samples_per_class, pca_enabled, fp_path, seed=seed)

        # Step 2: Prepare deployment artifacts
        prepare_deployment(model_dir, pca_enabled, pca_components,
                           samples_per_class, experiment_id, seed=seed,
                           test_images_per_class=test_per_class)

        # Step 3: Subscribe → Deploy → Wait (all in one async call)
        started_at = time.time()
        result_csv = collect_result(nats_url, device_id, experiment_id,
                                    result_timeout, results_dir, power_cfg)

        # Step 4: Save metadata
        pca_explained_variance = None
        if pca_enabled and fp_path and fp_path.exists():
            import numpy as np
            npz = np.load(fp_path, allow_pickle=False)
            if "explained_variance" in npz:
                pca_explained_variance = float(npz["explained_variance"])

        status = "TIMEOUT"
        if result_csv:
            # Energy/power metrics from the external Voltcraft trace
            power_metrics = {}
            if power_cfg.get("enabled"):
                n_images = _count_inferences(result_csv)
                power_metrics = compute_power_metrics(
                    result_csv.parent, n_images=n_images, baseline_power_w=baseline_w,
                )
                inf = power_metrics.get("inference_window", {})
                if inf:
                    print(f"[POWER] avg={inf.get('avg_power_W')} W  "
                          f"E={inf.get('energy_J')} J  "
                          f"E/img={inf.get('energy_per_image_J')} J")

            meta_path = result_csv.parent / "experiment_metadata.json"
            with open(meta_path, "w") as f:
                json.dump({
                    **exp,
                    "experiment_id":           experiment_id,
                    "started_at":              datetime.fromtimestamp(started_at).isoformat(),
                    "result_received_at":      datetime.now().isoformat(),
                    "pca_explained_variance_%": pca_explained_variance,
                    "power":                   power_metrics,
                }, f, indent=2)
            print(f"[{idx}/{total}]  Saved to {result_csv.parent.relative_to(PROJECT_ROOT)}")
            status = "OK"

        summary.append({"experiment_id": experiment_id, "status": status})
        print(f"[{idx}/{total}]  {experiment_id}  →  {status}")

    # Write summary
    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "total":   total,
            "ok":      sum(1 for s in summary if s["status"] == "OK"),
            "timeout": sum(1 for s in summary if s["status"] == "TIMEOUT"),
            "experiments": summary,
        }, f, indent=2)

    ok_count = sum(1 for s in summary if s["status"] == "OK")
    print(f"\n{'='*60}")
    print(f"Pipeline complete: {ok_count}/{total} experiments succeeded.")
    print(f"Results  : {results_dir}")
    print(f"Summary  : {summary_path.name}")
    print(f"{'='*60}")


def measure_baseline(cfg: Dict, seconds: float) -> None:
    """Log idle Pi power for `seconds` and print the mean — gives P_idle for net energy.

    Run this once with the Pi booted and deployment-receiver listening, but NOT
    running any inference. Put the printed value into power.baseline_power_w.
    """
    power_cfg = cfg.get("power", {})
    port = power_cfg.get("port", "COM3")
    rate = power_cfg.get("rate", 5.0)
    out_csv = ensure_dir(DEFAULT_RESULTS_DIR) / "baseline_power.csv"

    from dsp3005_reader import PowerLogger
    print(f"[BASELINE] Logging idle power on {port} @ {rate} Hz for {seconds}s → {out_csv}")
    logger = PowerLogger(out_csv, port=port, rate=rate)
    logger.start()
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[BASELINE] Interrupted early.")
    logger.stop()

    ts, pw = _read_power_csv(out_csv)
    if pw:
        mean_w = sum(pw) / len(pw)
        print(f"\n[BASELINE] P_idle = {mean_w:.4f} W  (n={len(pw)} samples)")
        print(f"[BASELINE] Set  \"baseline_power_w\": {mean_w:.4f}  in benchmark_config.json")
    else:
        print(f"[BASELINE] No samples captured. Check device/port. error={logger.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN Sweet Spot Benchmark Pipeline")
    parser.add_argument("--config",      type=Path, default=DEFAULT_CONFIG,
                        help="Path to benchmark_config.json")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory to store results")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print planned experiments without executing")
    parser.add_argument("--measure-baseline", type=float, metavar="SECONDS",
                        help="Measure idle Pi power for N seconds and exit (gives P_idle)")
    parser.add_argument("--clean-results", action="store_true",
                        help="Delete all previous exp_* result folders before the sweep "
                             "(keeps results/analysis/ and the baseline). Use for a fresh run.")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    cfg = load_benchmark_config(args.config)

    if args.measure_baseline:
        measure_baseline(cfg, args.measure_baseline)
        return

    ensure_dir(args.results_dir)
    run_pipeline(cfg, args.results_dir, dry_run=args.dry_run, clean=args.clean_results)


if __name__ == "__main__":
    main()
