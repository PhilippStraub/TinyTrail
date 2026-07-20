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
  python benchmark_pipeline.py --rebuild-summary      # rebuild summaries from finished runs
  python benchmark_pipeline.py --measure-baseline 60  # measure idle Pi power, then exit
"""

import argparse
import asyncio
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile as zf_module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Project paths  (all relative to this file: server/benchmark/)
# ──────────────────────────────────────────────────────────────────────────────
BENCHMARK_DIR   = Path(__file__).parent
SERVER_DIR      = BENCHMARK_DIR.parent
PROJECT_ROOT    = SERVER_DIR.parent

RETRAINER_ROOT      = PROJECT_ROOT / "server/Retrainer/on-device-drift-detection-main"
DEFAULT_SCENARIO_NAME = "scenario_cifar10_nearest_centroid"
CONFIG_TEMPLATE_DIR   = BENCHMARK_DIR / "config_templates"

PACKAGER_MAIN   = PROJECT_ROOT / "server/Deployment-Packager/nats-model-service/src/nats_model_service/main.py"

DEFAULT_CONFIG      = BENCHMARK_DIR / "benchmark_config.json"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


@dataclass(frozen=True)
class ScenarioContext:
    name: str
    dataset_name: str
    py_example_dir: Path
    model_training_dir: Path
    raw_data_dir: Path
    model_dir: Path
    test_data_dir: Optional[Path]
    config_source: Path


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


def _safe_rmtree(path: Path) -> None:
    """Remove a directory only if it is inside the project workspace."""
    resolved = path.resolve()
    project = PROJECT_ROOT.resolve()
    if resolved != project and project not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside project root: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def _copy_tree_clean(src: Path, dst: Path) -> None:
    """Copy a Python package tree into raw_data without stale bytecode/cache files."""
    if not src.is_dir():
        raise FileNotFoundError(f"Required directory not found: {src}")
    if dst.exists():
        _safe_rmtree(dst)

    def ignore_cache(_dir: str, names: List[str]) -> List[str]:
        return [
            name for name in names
            if name == "__pycache__" or name.endswith(".pyc") or name == ".DS_Store"
        ]

    shutil.copytree(src, dst, ignore=ignore_cache)


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _scenario_specs_from_config(cfg: Dict) -> List[Dict]:
    """Return scenario specs from benchmark_config, accepting old and new keys."""
    scenario_keys = (
        "experiment_names",
        "experiment_name",
        "Experiment Name",
        "Experiment Names",
        "scenario_names",
        "scenarios",
        "scenario",
    )
    raw_value = None
    for key in scenario_keys:
        if key in cfg:
            raw_value = cfg[key]
            break

    raw_specs = _as_list(raw_value) or [DEFAULT_SCENARIO_NAME]
    specs: List[Dict] = []
    for item in raw_specs:
        if isinstance(item, str):
            specs.append({"name": item})
        elif isinstance(item, dict):
            name = item.get("name") or item.get("experiment_name") or item.get("Experiment Name")
            if not name:
                raise ValueError(f"Scenario spec is missing a name: {item}")
            specs.append({**item, "name": name})
        else:
            raise TypeError(f"Unsupported scenario spec: {item!r}")
    return specs


def _scenario_config(base_cfg: Dict, scenario_spec: Dict) -> Dict:
    """Merge optional per-scenario overrides over the global benchmark config."""
    scenario_cfg = dict(base_cfg)
    for key, value in scenario_spec.items():
        if key not in {"name", "experiment_name", "Experiment Name"}:
            scenario_cfg[key] = value
    return scenario_cfg


def _infer_dataset_name(scenario_name: str) -> str:
    lower = scenario_name.lower()
    if "cifar10" in lower:
        return "cifar10"
    if "imagenette" in lower:
        return "imagenette"
    if "imagenet" in lower:
        return "imagenet"
    if "banknote" in lower:
        return "banknote"
    parts = [p for p in lower.split("_") if p and p != "scenario"]
    return parts[0] if parts else "unknown"


def _resolve_test_data_dir(cfg: Dict, scenario_name: str, dataset_name: str) -> Optional[Path]:
    mapping = cfg.get("test_data_dirs", {})
    if isinstance(mapping, dict):
        mapped = mapping.get(scenario_name) or mapping.get(dataset_name)
        if mapped:
            return _project_path(mapped)

    if cfg.get("test_data_dir"):
        return _project_path(cfg["test_data_dir"])

    if dataset_name == "cifar10":
        return RETRAINER_ROOT / "examples/dataset_cifar10/test"
    if dataset_name == "imagenet":
        imagenet_test = RETRAINER_ROOT / "examples/dataset_imagenet/test"
        if imagenet_test.exists():
            return imagenet_test
        return RETRAINER_ROOT / "examples/dataset_imagenette/test"
    if dataset_name == "imagenette":
        return RETRAINER_ROOT / "examples/dataset_imagenette/test"
    return None


def _resolve_config_source(py_example_dir: Path, raw_data_dir: Path, scenario_name: str) -> Path:
    candidates = [
        CONFIG_TEMPLATE_DIR / f"{scenario_name}.json",
        py_example_dir / "config.json",
        raw_data_dir / "config.json",
        BENCHMARK_DIR / "config_template.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No config template found for {scenario_name}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def build_scenario_context(scenario_spec: Dict, base_cfg: Dict) -> Tuple[ScenarioContext, Dict]:
    scenario_name = str(scenario_spec["name"])
    scenario_cfg = _scenario_config(base_cfg, scenario_spec)
    dataset_name = str(scenario_cfg.get("dataset_name") or _infer_dataset_name(scenario_name))

    py_example_dir = RETRAINER_ROOT / f"examples/python_examples/{scenario_name}"
    model_training_dir = RETRAINER_ROOT / f"examples/model_training/{scenario_name}"
    raw_data_dir = PROJECT_ROOT / f"server/Deployment-Packager/nats-model-service/raw_data/{scenario_name}"

    if not py_example_dir.is_dir():
        raise FileNotFoundError(f"Python scenario not found: {py_example_dir}")

    ctx = ScenarioContext(
        name=scenario_name,
        dataset_name=dataset_name,
        py_example_dir=py_example_dir,
        model_training_dir=model_training_dir,
        raw_data_dir=raw_data_dir,
        model_dir=py_example_dir / "model",
        test_data_dir=_resolve_test_data_dir(scenario_cfg, scenario_name, dataset_name),
        config_source=_resolve_config_source(py_example_dir, raw_data_dir, scenario_name),
    )
    return ctx, scenario_cfg


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _load_base_config(ctx: ScenarioContext) -> Dict:
    return _load_json(ctx.config_source)


def _classifier_type(config: Dict) -> str:
    return str(config.get("tinylcm", {}).get("adaptive_classifier", {}).get("type", "unknown"))


def _inference_resolution(config: Dict) -> Optional[List[int]]:
    value = config.get("image_input", {}).get("inference_resolution")
    return list(value) if isinstance(value, list) else None


def _config_relpath(path_value: Optional[str], default: str) -> Path:
    raw = path_value or default
    raw = raw[2:] if raw.startswith("./") else raw
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Config artifact path must be relative inside the scenario: {raw}")
    return path


def _state_relpaths(config: Dict) -> Tuple[Path, Optional[Path]]:
    tinylcm = config.get("tinylcm", {})
    classifier = tinylcm.get("adaptive_classifier", {})
    initial_state = _config_relpath(
        classifier.get("initial_state_path"),
        "initial_state/knn_initial_state_objects.json",
    )

    reference_stats: Optional[Path] = None
    detectors = tinylcm.get("drift_detectors", [])
    if detectors:
        reference_stats = _config_relpath(
            detectors[0].get("reference_stats_path"),
            "initial_state/knn_reference_stats.json",
        )
    return initial_state, reference_stats


def scenario_results_dir(results_dir: Path, ctx: ScenarioContext) -> Path:
    return results_dir / ctx.name


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_benchmark_config(config_path: Path) -> Dict:
    with open(config_path) as f:
        return json.load(f)


def _pca_target_from_config(pca_cfg: Dict) -> Tuple[Optional[float | int], Optional[str]]:
    if not pca_cfg.get("enabled", True):
        return None, None

    variance = pca_cfg.get("explained_variance") if "explained_variance" in pca_cfg else pca_cfg.get("variance")
    if variance is not None:
        target = float(variance)
        if target > 1.0:
            target = target / 100.0
        if not 0.0 < target < 1.0:
            raise ValueError(f"explained_variance must be in (0, 1) or (0, 100), got {variance}")
        return target, "explained_variance"

    if "components" not in pca_cfg:
        raise ValueError("Enabled PCA config needs either 'components' or 'explained_variance'.")
    return int(pca_cfg["components"]), "components"


def _pca_label(target: Optional[float | int], target_type: Optional[str]) -> str:
    if target is None:
        return "nopca"
    if target_type == "explained_variance":
        pct = float(target) * 100.0
        return f"pca{f'{pct:g}'.replace('.', 'p')}var"
    return f"pca{int(target)}"


def _feature_processor_cache_name(pca_target: float | int, pca_label: Optional[str] = None) -> str:
    if pca_label is None:
        target_type = "explained_variance" if isinstance(pca_target, float) and 0.0 < pca_target < 1.0 else "components"
        pca_label = _pca_label(pca_target, target_type)
    return f"feature_processor_{pca_label}.npz"


def build_experiment_list(cfg: Dict) -> List[Dict]:
    """Expand benchmark_config into a flat ordered list of experiments.

    Order: pca_config → samples_per_class → seed
    This groups all seeds for one (pca, spc) pair together so variance
    across seeds is immediately visible in the results directory.
    """
    experiments = []
    seeds = cfg.get("seeds", [42])
    for pca_cfg in cfg["pca_configs"]:
        pca_enabled = pca_cfg.get("enabled", True)
        pca_target, pca_target_type = _pca_target_from_config(pca_cfg)
        pca_label = _pca_label(pca_target, pca_target_type)
        for spc in cfg["samples_per_class"]:
            for seed in seeds:
                experiments.append({
                    "pca_enabled":       pca_enabled,
                    "pca_components":    pca_target if pca_target_type == "components" else None,
                    "pca_target":        pca_target,
                    "pca_target_type":   pca_target_type,
                    "pca_label":         pca_label,
                    "samples_per_class": spc,
                    "seed":              seed,
                })
    return experiments


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 – Feature processor training
# ──────────────────────────────────────────────────────────────────────────────

def _feature_processor_output_dim(path: Path) -> Optional[int]:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as npz:
            if "output_dim" in npz:
                return int(npz["output_dim"])
            if "W" in npz:
                return int(npz["W"].shape[0])
    except Exception:
        return None
    return None


def _feature_processor_explained_variance(path: Path) -> Optional[float]:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as npz:
            if "explained_variance" in npz:
                return float(npz["explained_variance"])
    except Exception:
        return None
    return None


def _find_make_feature_processor(ctx: ScenarioContext) -> Optional[Path]:
    exact = ctx.model_training_dir / "make_feature_processor.py"
    if exact.exists():
        return exact

    training_root = RETRAINER_ROOT / "examples/model_training"
    candidates = sorted(
        p / "make_feature_processor.py"
        for p in training_root.glob(f"*{ctx.dataset_name}*")
        if (p / "make_feature_processor.py").exists()
    )
    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if p.parent.name.startswith("scenario_") else 1,
            p.parent.name,
        ),
    )
    return candidates[0] if candidates else None


def _default_train_data_dir(ctx: ScenarioContext) -> Optional[Path]:
    if ctx.dataset_name == "cifar10":
        return RETRAINER_ROOT / "examples/dataset_cifar10/train"
    if ctx.dataset_name == "imagenet":
        imagenet_train = RETRAINER_ROOT / "examples/dataset_imagenet/train"
        if imagenet_train.exists():
            return imagenet_train
        return RETRAINER_ROOT / "examples/dataset_imagenette/train"
    if ctx.dataset_name == "imagenette":
        return RETRAINER_ROOT / "examples/dataset_imagenette/train"
    return None


def _feature_processor_img_size(config: Dict) -> Optional[int]:
    resolution = _inference_resolution(config)
    if not resolution:
        return None
    if len(resolution) < 2 or resolution[0] != resolution[1]:
        raise ValueError(f"Feature processor expects square inference_resolution, got: {resolution}")
    return int(resolution[0])


def ensure_feature_processor(ctx: ScenarioContext, pca_components: int, config: Optional[Dict] = None) -> Path:
    """Return path to the cached feature_processor_pca{N}.npz, training it if needed."""
    model_dir = ctx.model_dir
    cached = model_dir / f"feature_processor_pca{pca_components}.npz"
    if cached.exists():
        print(f"[FP] feature_processor_pca{pca_components}.npz already cached — skipping.")
        return cached

    print(f"\n[FP] Training feature processor (PCA components = {pca_components}) ...")
    make_fp = _find_make_feature_processor(ctx)
    if make_fp is None:
        default_fp = model_dir / "feature_processor.npz"
        if default_fp.exists() and _feature_processor_output_dim(default_fp) == pca_components:
            shutil.copy2(default_fp, cached)
            print(f"[FP:{ctx.name}] Reused existing feature_processor.npz as {cached.name}.")
            return cached
        raise FileNotFoundError(
            f"No make_feature_processor.py found for {ctx.name}, and no reusable "
            f"feature_processor.npz with output_dim={pca_components} exists."
        )
    config = config or _load_base_config(ctx)
    cmd = [
        sys.executable,
        make_fp,
        "--pca-components",
        str(pca_components),
        "--model-path",
        model_dir / "model_object.tflite",
        "--output-dir",
        model_dir,
    ]
    train_data_dir = _default_train_data_dir(ctx)
    if train_data_dir is not None and train_data_dir.exists():
        cmd.extend(["--data-dir", train_data_dir])
    img_size = _feature_processor_img_size(config)
    if img_size is not None:
        cmd.extend(["--img-size", str(img_size)])
    run_subprocess(
        cmd,
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

def ensure_feature_processor(
    ctx: ScenarioContext,
    pca_target: float | int,
    config: Optional[Dict] = None,
    pca_label: Optional[str] = None,
) -> Path:
    """Return the cached PCA processor, training it when needed."""
    model_dir = ctx.model_dir
    cached = model_dir / _feature_processor_cache_name(pca_target, pca_label)
    if cached.exists():
        print(f"[FP] {cached.name} already cached - skipping.")
        return cached

    variance_target = isinstance(pca_target, float) and 0.0 < pca_target < 1.0
    if variance_target:
        print(f"\n[FP] Training feature processor (PCA explained variance = {pca_target * 100:.1f}%) ...")
    else:
        print(f"\n[FP] Training feature processor (PCA components = {pca_target}) ...")

    default_fp = model_dir / "feature_processor.npz"
    if default_fp.exists():
        if variance_target:
            explained = _feature_processor_explained_variance(default_fp)
            if explained is not None and explained >= float(pca_target) * 100.0 - 1e-6:
                shutil.copy2(default_fp, cached)
                print(f"[FP:{ctx.name}] Reused existing feature_processor.npz as {cached.name}.")
                return cached
        elif _feature_processor_output_dim(default_fp) == int(pca_target):
            shutil.copy2(default_fp, cached)
            print(f"[FP:{ctx.name}] Reused existing feature_processor.npz as {cached.name}.")
            return cached

    make_fp = _find_make_feature_processor(ctx)
    if make_fp is None:
        raise FileNotFoundError(
            f"No make_feature_processor.py found for {ctx.name}, and no reusable "
            f"feature_processor.npz for PCA target={pca_target} exists."
        )

    config = config or _load_base_config(ctx)
    cmd = [
        sys.executable,
        make_fp,
        "--pca-components",
        str(pca_target),
        "--model-path",
        model_dir / "model_object.tflite",
        "--output-dir",
        model_dir,
    ]
    train_data_dir = _default_train_data_dir(ctx)
    if train_data_dir is not None and train_data_dir.exists():
        cmd.extend(["--data-dir", train_data_dir])
    img_size = _feature_processor_img_size(config)
    if img_size is not None:
        cmd.extend(["--img-size", str(img_size)])

    run_subprocess(
        cmd,
        label=f"make_feature_processor(pca={pca_label or pca_target})",
    )

    default_fp = model_dir / "feature_processor.npz"
    if not default_fp.exists():
        raise FileNotFoundError(f"Expected {default_fp} after training - not found.")
    shutil.copy2(default_fp, cached)
    print(f"[FP] Cached as {cached.name}")
    return cached


_cis_modules: Dict[str, Any] = {}

def _get_cis_module(ctx: ScenarioContext):
    """Import the scenario's create_initial_state once and cache it.

    The first call pays the TensorFlow import (~9 s) a single time for the whole
    sweep; every later experiment reuses the module — and its cached TFLite
    interpreter — in-process, with no per-experiment subprocess or re-import.
    """
    if ctx.name not in _cis_modules:
        module_path = ctx.py_example_dir / "create_initial_state.py"
        if not module_path.exists():
            raise FileNotFoundError(f"create_initial_state.py not found: {module_path}")
        if str(ctx.py_example_dir) not in sys.path:
            sys.path.insert(0, str(ctx.py_example_dir))
        # In-process prints from create_initial_state may contain non-cp1252 chars;
        # make stdout/stderr UTF-8 so a Windows console can't crash on them (the
        # subprocess path used to get this via PYTHONUTF8).
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
        module_name = f"benchmark_create_initial_state_{_safe_slug(ctx.name)}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _cis_modules[ctx.name] = module
    return _cis_modules[ctx.name]


def create_initial_state(
    ctx: ScenarioContext,
    samples_per_class: int,
    pca_enabled: bool,
    feature_processor_path: Optional[Path],
    seed: int = 42,
) -> None:
    # In-process call (was a subprocess) so the whole sweep pays the TF import once.
    cis = _get_cis_module(ctx)
    print(f"\n[create_initial_state:{ctx.name}(spc={samples_per_class},pca={pca_enabled},seed={seed})] (in-process)")
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

def _load_legacy_base_config() -> Dict:
    template = BENCHMARK_DIR / "config_template.json"
    if not template.exists():
        src = build_scenario_context({"name": DEFAULT_SCENARIO_NAME}, {})[0].raw_data_dir / "config.json"
        shutil.copy2(src, template)
        print(f"[DEPLOY] Saved config_template.json from {src.relative_to(PROJECT_ROOT)}")
    with open(template) as f:
        return json.load(f)


def populate_test_images(
    ctx: ScenarioContext,
    config: Dict,
    seed: int,
    per_class: int,
) -> int:
    """Rebuild raw_data/input_images from the held-out TEST split, seeded.

    Evaluates on novelty (never-seen) data, and — because the draw depends on the
    seed — different seeds cover different test images, expanding effective test
    coverage across the sweep. Returns the total number of images placed.
    """
    import random

    dst_dir = ctx.raw_data_dir / "input_images"
    if dst_dir.exists():
        _safe_rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if ctx.test_data_dir is None or not ctx.test_data_dir.is_dir():
        raise FileNotFoundError(
            f"Test split not found for {ctx.name}: {ctx.test_data_dir}\n"
            f"Set test_data_dir/test_data_dirs in benchmark_config.json or export the dataset split."
        )

    supported_exts = {
        str(ext).lower()
        for ext in config.get("image_input", {}).get(
            "supported_extensions",
            [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        )
    }
    total = 0
    for ci, class_dir in enumerate(sorted(p for p in ctx.test_data_dir.iterdir() if p.is_dir())):
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts)
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


def _copy_static_scenario_artifacts(ctx: ScenarioContext) -> None:
    ensure_dir(ctx.raw_data_dir)
    dst_model = ensure_dir(ctx.raw_data_dir / "model")
    ensure_dir(ctx.raw_data_dir / "initial_state")
    for stale_cache in dst_model.glob("feature_processor_pca*.npz"):
        stale_cache.unlink()

    for name in ("main.py", "create_initial_state.py"):
        src = ctx.py_example_dir / name
        if src.exists():
            shutil.copy2(src, ctx.raw_data_dir / name)

    shared_package_dirs = (
        (RETRAINER_ROOT / "examples" / "utils", ctx.raw_data_dir / "utils"),
        (RETRAINER_ROOT / "tinylcm_python", ctx.raw_data_dir / "tinylcm_python"),
    )
    for src, dst in shared_package_dirs:
        _copy_tree_clean(src, dst)

    if not ctx.model_dir.is_dir():
        raise FileNotFoundError(f"Scenario model directory not found: {ctx.model_dir}")
    for src in ctx.model_dir.iterdir():
        if src.is_file():
            if src.name.startswith("feature_processor_pca") and src.suffix == ".npz":
                continue
            shutil.copy2(src, dst_model / src.name)


def _copy_state_artifacts(ctx: ScenarioContext, config: Dict) -> None:
    initial_state_rel, reference_stats_rel = _state_relpaths(config)
    for rel_path in (initial_state_rel, reference_stats_rel):
        if rel_path is None:
            continue
        src = ctx.py_example_dir / rel_path
        dst = ctx.raw_data_dir / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Expected scenario artifact not found: {src}")
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def _num_classes(ctx: ScenarioContext) -> int:
    """Class count for sizing the adaptive classifier's max_samples.

    Reads labels_object.txt — always present regardless of drift-detection config —
    instead of drift_detectors[0]['stable_known_classes'], which doesn't exist once
    drift detection is disabled ("drift_detectors": []) and would raise IndexError.
    """
    labels_path = ctx.model_dir / "labels_object.txt"
    labels = [ln.strip() for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return len(labels) if labels else 10


def prepare_deployment(
    ctx: ScenarioContext,
    pca_enabled: bool,
    pca_components: Optional[float | int],
    samples_per_class: int,
    experiment_id: str,
    seed: int = 42,
    test_images_per_class: int = 0,
    pca_label: Optional[str] = None,
) -> None:
    """Copy all artifacts to raw_data and write a patched config.json."""
    config = _load_base_config(ctx)
    _copy_static_scenario_artifacts(ctx)

    # Rebuild the test set from the held-out split (if configured)
    if test_images_per_class and test_images_per_class > 0:
        n = populate_test_images(ctx, config, seed, test_images_per_class)
        print(f"[DEPLOY:{ctx.name}] input_images: {n} test images "
              f"({test_images_per_class}/class, seed={seed}) from TEST split")

    if pca_enabled and pca_components:
        fp_src = ctx.model_dir / _feature_processor_cache_name(pca_components, pca_label)
        shutil.copy2(fp_src, ctx.raw_data_dir / "model" / "feature_processor.npz")

    _copy_state_artifacts(ctx, config)

    if "feature_transformation" in config.get("tinylcm", {}):
        config["tinylcm"]["feature_transformation"]["enabled"] = pca_enabled

    num_classes = _num_classes(ctx)
    config["tinylcm"]["adaptive_classifier"]["max_samples"] = samples_per_class * num_classes

    # Same seed for test-image shuffle so each run is reproducible end-to-end
    if "image_input" in config:
        config["image_input"]["shuffle_seed"] = seed

    with open(ctx.raw_data_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(ctx.raw_data_dir / "experiment_config.json", "w") as f:
        json.dump({
            "experiment_id":         experiment_id,
            "scenario_name":         ctx.name,
            "dataset_name":          ctx.dataset_name,
            "classifier_type":       _classifier_type(config),
            "config_source":         str(ctx.config_source),
            "inference_resolution":  _inference_resolution(config),
            "pca_enabled":           pca_enabled,
            "pca_components":        pca_components,
            "samples_per_class":     samples_per_class,
            "seed":                  seed,
            "test_images_per_class": test_images_per_class,
            "deployed_at":           datetime.now().isoformat(),
        }, f, indent=2)

    print(f"[DEPLOY:{ctx.name}] Artifacts ready for {experiment_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 – Deploy + collect result via NATS (no Analytical-Data-Processor needed)
# ──────────────────────────────────────────────────────────────────────────────

async def _subscribe_deploy_collect(
    nats_url: str,
    device_id: str,
    scenario_name: str,
    experiment_id: str,
    timeout_seconds: int,
    results_dir: Path,
    power_cfg: Optional[Dict] = None,
) -> Optional[Path]:
    """
    1. Subscribe to analysis.logs.{device_id} (deliver_policy=new)
    2. Deploy in a thread executor so the event loop stays alive
    3. Wait for a ZIP containing merged_analysis_{experiment_id}*.csv
    4. Save all files from the ZIP into results_dir/{scenario_name}/{experiment_id}/
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
    print(f"[RESULT] Subscribed. Deploying {scenario_name}/{experiment_id} ...")

    # Start fresh: wipe any artifacts from a previous run of this experiment id, so
    # the folder only ever holds one run's files. Otherwise timestamped logs from
    # earlier runs accumulate and the analysis can't tell which run is current.
    exp_dir = results_dir / experiment_id
    if exp_dir.exists():
        _safe_rmtree(exp_dir)
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
            msg = f"[POWER] Could not start power logger: {e}"
            if power_cfg.get("required", False):
                raise RuntimeError(msg) from e
            print(f"{msg} — continuing without power.")
            power_logger = None

    # Deploy in executor — subprocess.run() is blocking, but runs in a thread
    # so the asyncio event loop continues processing incoming NATS messages
    deploy_env = {**os.environ, "NATS_URL": nats_url}
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [str(sys.executable), str(PACKAGER_MAIN), scenario_name],
            env=deploy_env,
            check=True,
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
        if power_logger.error and power_cfg.get("required", False):
            raise RuntimeError(f"Power logging failed during {experiment_id}: {power_logger.error}")

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
    scenario_name: str,
    experiment_id: str,
    timeout_seconds: int,
    results_dir: Path,
    power_cfg: Optional[Dict] = None,
) -> Optional[Path]:
    """Synchronous wrapper — runs the async subscribe+deploy+collect in a fresh event loop."""
    return asyncio.run(_subscribe_deploy_collect(
        nats_url, device_id, scenario_name, experiment_id, timeout_seconds, results_dir, power_cfg,
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

def _legacy_clean_experiment_results(results_dir: Path) -> int:
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


def _legacy_run_pipeline(cfg: Dict, results_dir: Path, dry_run: bool = False, clean: bool = False) -> None:
    device_id      = cfg["device_id"]
    nats_url       = cfg.get("nats_url", "nats://localhost:4222")
    result_timeout = cfg.get("result_timeout_seconds", 1800)
    model_dir      = build_scenario_context({"name": DEFAULT_SCENARIO_NAME}, cfg)[0].model_dir
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


def clean_experiment_results(results_dir: Path) -> int:
    """Delete every per-experiment folder (exp_*) from one result directory."""
    if not results_dir.exists():
        return 0
    exp_dirs = sorted(p for p in results_dir.glob("exp_*") if p.is_dir())
    for p in exp_dirs:
        _safe_rmtree(p)
    print(f"[CLEAN] {results_dir}: removed {len(exp_dirs)} previous experiment folder(s).")
    return len(exp_dirs)


def _scenario_runs(cfg: Dict) -> List[Tuple[ScenarioContext, Dict, List[Dict], Dict]]:
    runs: List[Tuple[ScenarioContext, Dict, List[Dict], Dict]] = []
    for spec in _scenario_specs_from_config(cfg):
        ctx, scenario_cfg = build_scenario_context(spec, cfg)
        experiments = build_experiment_list(scenario_cfg)
        base_config = _load_base_config(ctx)
        runs.append((ctx, scenario_cfg, experiments, base_config))
    return runs


def _write_summary(summary_path: Path, summary: List[Dict], extra: Optional[Dict] = None) -> None:
    ensure_dir(summary_path.parent)
    with open(summary_path, "w") as f:
        json.dump({
            **(extra or {}),
            "completed_at": datetime.now().isoformat(),
            "total":   len(summary),
            "ok":      sum(1 for s in summary if s["status"] == "OK"),
            "timeout": sum(1 for s in summary if s["status"] == "TIMEOUT"),
            "experiments": summary,
        }, f, indent=2)


def _experiment_id(index: int, exp: Dict) -> str:
    return f"exp_{index+1:03d}_{exp['pca_label']}_spc{exp['samples_per_class']}_s{exp['seed']}"


def _is_completed_experiment(exp_dir: Path) -> bool:
    """A run counts as completed only once metadata and merged analysis exist."""
    if not exp_dir.is_dir():
        return False
    if not (exp_dir / "experiment_metadata.json").exists():
        return False
    return any(exp_dir.glob("merged_analysis*.csv"))


def _completed_summary_entry(
    ctx: ScenarioContext,
    classifier: str,
    experiment_id: str,
    exp_dir: Path,
) -> Dict:
    meta = {}
    meta_path = exp_dir / "experiment_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return {
        "scenario_name": meta.get("scenario_name", ctx.name),
        "dataset_name": meta.get("dataset_name", ctx.dataset_name),
        "classifier_type": meta.get("classifier_type", classifier),
        "experiment_id": meta.get("experiment_id", experiment_id),
        "status": "OK",
    }


def _load_completed_scenario_summary(
    ctx: ScenarioContext,
    scenario_results: Path,
    experiments: List[Dict],
    classifier: str,
) -> Tuple[List[Dict], set]:
    """Return OK summary entries for expected experiments already on disk."""
    summary: List[Dict] = []
    completed_ids = set()
    for i, exp in enumerate(experiments):
        experiment_id = _experiment_id(i, exp)
        exp_dir = scenario_results / experiment_id
        if not _is_completed_experiment(exp_dir):
            continue
        summary.append(_completed_summary_entry(ctx, classifier, experiment_id, exp_dir))
        completed_ids.add(experiment_id)
    return summary, completed_ids


def rebuild_existing_summaries(cfg: Dict, results_dir: Path) -> None:
    """Recreate benchmark_summary.json files from complete result folders."""
    runs = _scenario_runs(cfg)
    all_summary: List[Dict] = []
    scenarios: List[str] = []

    for ctx, _scenario_cfg, experiments, base_config in runs:
        classifier = _classifier_type(base_config)
        resolution = _inference_resolution(base_config)
        scenario_results = results_dir / ctx.name
        summary, _completed_ids = _load_completed_scenario_summary(
            ctx, scenario_results, experiments, classifier
        )
        if not scenario_results.exists() and not summary:
            continue

        _write_summary(
            scenario_results / "benchmark_summary.json",
            summary,
            extra={
                "scenario_name": ctx.name,
                "dataset_name": ctx.dataset_name,
                "classifier_type": classifier,
                "config_source": str(ctx.config_source),
                "inference_resolution": resolution,
            },
        )
        print(f"[SUMMARY] {ctx.name}: {len(summary)} completed OK run(s)")
        all_summary.extend(summary)
        scenarios.append(ctx.name)

    _write_summary(
        results_dir / "benchmark_summary.json",
        all_summary,
        extra={"scenarios": scenarios},
    )
    print(f"[SUMMARY] root: {len(all_summary)} completed OK run(s)")


def _run_scenario(
    ctx: ScenarioContext,
    scenario_cfg: Dict,
    experiments: List[Dict],
    base_config: Dict,
    scenario_results: Path,
    device_id: str,
    nats_url: str,
    result_timeout: int,
    power_cfg: Dict,
    baseline_w: Optional[float],
    dry_run: bool,
    clean: bool,
) -> List[Dict]:
    total = len(experiments)
    test_per_class = scenario_cfg.get("test_images_per_class", 0)
    abort_on_timeout = scenario_cfg.get("abort_on_timeout", True)
    skip_completed = scenario_cfg.get("skip_completed", True)
    classifier = _classifier_type(base_config)
    resolution = _inference_resolution(base_config)

    print(f"\n{'='*60}")
    print(f"Scenario : {ctx.name}")
    print(f"Dataset  : {ctx.dataset_name}")
    print(f"Classifier: {classifier}")
    print(f"Config   : {ctx.config_source.relative_to(PROJECT_ROOT)}")
    print(f"Results  : {scenario_results}")
    if resolution:
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
    if test_per_class:
        print(f"Test     : {test_per_class} images/class from {ctx.test_data_dir}")
    print(f"Runs     : {total}")
    print(f"{'='*60}")

    if dry_run:
        completed_ids = set()
        if skip_completed:
            _summary, completed_ids = _load_completed_scenario_summary(
                ctx, scenario_results, experiments, classifier
            )
        for i, exp in enumerate(experiments):
            experiment_id = _experiment_id(i, exp)
            state = "SKIP" if experiment_id in completed_ids else "RUN "
            print(
                f"  [{i+1:03d}] {state} {ctx.name}  {exp['pca_label']}  "
                f"spc={exp['samples_per_class']}  seed={exp['seed']}"
            )
        return [
            {
                "scenario_name": ctx.name,
                "experiment_id": _experiment_id(i, exp),
                "status": "SKIP" if _experiment_id(i, exp) in completed_ids else "DRY_RUN",
            }
            for i, exp in enumerate(experiments)
        ]

    ensure_dir(scenario_results)
    if clean:
        clean_experiment_results(scenario_results)

    needed_pca = {
        (exp["pca_label"], exp["pca_target"])
        for exp in experiments
        if exp["pca_enabled"] and exp.get("pca_target") is not None
    }
    for pca_label, pca_target in sorted(needed_pca):
        ensure_feature_processor(ctx, pca_target, base_config, pca_label=pca_label)

    summary: List[Dict] = []
    completed_ids = set()
    if skip_completed:
        summary, completed_ids = _load_completed_scenario_summary(
            ctx, scenario_results, experiments, classifier
        )
        if completed_ids:
            print(f"[RESUME] {len(completed_ids)} completed OK run(s) will be skipped.")
            _write_summary(
                scenario_results / "benchmark_summary.json",
                summary,
                extra={
                    "scenario_name": ctx.name,
                    "dataset_name": ctx.dataset_name,
                    "classifier_type": classifier,
                    "config_source": str(ctx.config_source),
                    "inference_resolution": resolution,
                    "partial": True,
                },
            )

    for i, exp in enumerate(experiments):
        pca_enabled       = exp["pca_enabled"]
        pca_target        = exp.get("pca_target", exp["pca_components"])
        pca_components    = pca_target
        samples_per_class = exp["samples_per_class"]
        pca_label         = exp["pca_label"]
        seed              = exp["seed"]
        idx               = f"{i+1:03d}"
        experiment_id     = _experiment_id(i, exp)

        if experiment_id in completed_ids:
            print(f"[SKIP] {ctx.name}/{experiment_id} already has metadata + merged_analysis.")
            continue

        print(f"\n{'-'*60}")
        print(f"[{idx}/{total}]  {ctx.name}/{experiment_id}")
        print(f"  PCA  : enabled={pca_enabled}, target={pca_components}")
        print(f"  Samples/class: {samples_per_class}  Seed: {seed}")
        print(f"{'-'*60}")

        fp_path = (ctx.model_dir / _feature_processor_cache_name(pca_components, pca_label)
                   if pca_enabled and pca_components else None)

        create_initial_state(ctx, samples_per_class, pca_enabled, fp_path, seed=seed)

        prepare_deployment(ctx, pca_enabled, pca_components,
                           samples_per_class, experiment_id, seed=seed,
                           test_images_per_class=test_per_class,
                           pca_label=pca_label)

        started_at = time.time()
        result_csv = collect_result(nats_url, device_id, ctx.name, experiment_id,
                                    result_timeout, scenario_results, power_cfg)

        pca_explained_variance = None
        if pca_enabled and fp_path and fp_path.exists():
            import numpy as np
            with np.load(fp_path, allow_pickle=False) as npz:
                if "explained_variance" in npz:
                    pca_explained_variance = float(npz["explained_variance"])

        status = "TIMEOUT"
        if result_csv:
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
                    "experiment_id":            experiment_id,
                    "scenario_name":            ctx.name,
                    "dataset_name":             ctx.dataset_name,
                    "classifier_type":          classifier,
                    "config_source":            str(ctx.config_source),
                    "inference_resolution":     resolution,
                    "python_example_dir":       str(ctx.py_example_dir),
                    "raw_data_dir":             str(ctx.raw_data_dir),
                    "started_at":               datetime.fromtimestamp(started_at).isoformat(),
                    "result_received_at":       datetime.now().isoformat(),
                    "pca_explained_variance_%": pca_explained_variance,
                    "power":                    power_metrics,
                }, f, indent=2)
            print(f"[{idx}/{total}]  Saved to {result_csv.parent.relative_to(PROJECT_ROOT)}")
            status = "OK"

        summary.append({
            "scenario_name": ctx.name,
            "dataset_name": ctx.dataset_name,
            "classifier_type": classifier,
            "experiment_id": experiment_id,
            "status": status,
        })
        print(f"[{idx}/{total}]  {ctx.name}/{experiment_id}  ->  {status}")

        _write_summary(
            scenario_results / "benchmark_summary.json",
            summary,
            extra={
                "scenario_name": ctx.name,
                "dataset_name": ctx.dataset_name,
                "classifier_type": classifier,
                "config_source": str(ctx.config_source),
                "inference_resolution": resolution,
                "partial": True,
            },
        )

        if status == "TIMEOUT" and abort_on_timeout:
            raise RuntimeError(
                f"Aborting after timeout in {ctx.name}/{experiment_id}. "
                "No result ZIP was received from the Pi."
            )

    _write_summary(
        scenario_results / "benchmark_summary.json",
        summary,
        extra={
            "scenario_name": ctx.name,
            "dataset_name": ctx.dataset_name,
            "classifier_type": classifier,
            "config_source": str(ctx.config_source),
            "inference_resolution": resolution,
        },
    )
    return summary


def run_pipeline(cfg: Dict, results_dir: Path, dry_run: bool = False, clean: bool = False) -> None:
    device_id      = cfg["device_id"]
    nats_url       = cfg.get("nats_url", "nats://localhost:4222")
    result_timeout = cfg.get("result_timeout_seconds", 1800)
    power_cfg      = cfg.get("power", {})
    baseline_w     = power_cfg.get("baseline_power_w")

    runs = _scenario_runs(cfg)
    total = sum(len(experiments) for _, _, experiments, _ in runs)

    print(f"\n{'='*60}")
    print(f"BENCHMARK PIPELINE  -  {len(runs)} scenario(s), {total} experiment(s)")
    print(f"Device : {device_id}")
    print(f"NATS   : {nats_url}")
    print(f"Results: {results_dir}")
    if power_cfg.get("enabled"):
        print(f"Power  : {power_cfg.get('port', 'COM3')} @ {power_cfg.get('rate', 5.0)} Hz"
              + (f"  (idle baseline {baseline_w} W)" if baseline_w is not None else ""))
    print(f"{'='*60}")

    if clean and not dry_run:
        clean_experiment_results(results_dir)

    all_summary: List[Dict] = []
    for ctx, scenario_cfg, experiments, base_config in runs:
        summary = _run_scenario(
            ctx=ctx,
            scenario_cfg=scenario_cfg,
            experiments=experiments,
            base_config=base_config,
            scenario_results=scenario_results_dir(results_dir, ctx),
            device_id=device_id,
            nats_url=nats_url,
            result_timeout=result_timeout,
            power_cfg=power_cfg,
            baseline_w=baseline_w,
            dry_run=dry_run,
            clean=clean,
        )
        all_summary.extend(summary)

    if dry_run:
        print("\n(dry-run: no experiments executed)")
        return

    summary_path = results_dir / "benchmark_summary.json"
    _write_summary(
        summary_path,
        all_summary,
        extra={
            "scenarios": [ctx.name for ctx, _, _, _ in runs],
            "result_layout": "results/<scenario_name>/exp_*",
        },
    )

    ok_count = sum(1 for s in all_summary if s["status"] == "OK")
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
    parser.add_argument("--rebuild-summary", action="store_true",
                        help="Rebuild benchmark_summary.json from complete exp_* folders and exit.")
    parser.add_argument("--rerun-completed", action="store_true",
                        help="Deploy runs again even if metadata + merged_analysis already exist.")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    cfg = load_benchmark_config(args.config)

    if args.measure_baseline:
        measure_baseline(cfg, args.measure_baseline)
        return

    if args.rerun_completed:
        cfg["skip_completed"] = False

    ensure_dir(args.results_dir)
    if args.rebuild_summary:
        rebuild_existing_summaries(cfg, args.results_dir)
        return

    run_pipeline(cfg, args.results_dir, dry_run=args.dry_run, clean=args.clean_results)


if __name__ == "__main__":
    main()
