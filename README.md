# TinyTrail Repository

# Process Overview

# Tool Archicture Overview


Detailed Architecture Overview:
<p align="center">
  <img src="./tinyTrail-detailed.png" alt="TinyTrail Tool Overview" width="600">
</p>

Subsequent folders explain the components inlcuding setup in detail.

# Want to add a new Use Case?

Adding a new use case means adding a new **dataset** end-to-end: a trained feature
extractor, one or more classifier scenarios built on it, and a benchmark entry that
measures it on the Raspberry Pi. Every dataset already in the repo (CIFAR-10,
Imagenette, Rover-Lab) was built this same way. The steps below only order and abstract
that recipe — the actual detail lives in the two component READMEs linked from each
step, so read those before starting, not this page alone.

## 1. Retrainer — dataset, feature extractor, classifier scenarios

Everything in this phase lives under
`server/Retrainer/on-device-drift-detection-main/examples/`. Full detail:
[`server/Retrainer/README.md`](server/Retrainer/README.md).

1. **Export the dataset** to per-class image folders
   (`examples/dataset_<name>/train/<class>/*`, `examples/dataset_<name>/test/<class>/*`),
   using `examples/dataset_cifar10/export_cifar10.py` as a template.
2. **Choose a feature-extractor backbone** matching the dataset's resolution and
   complexity — see
   [Feature Extractor Architectures Per Dataset](server/Retrainer/README.md#feature-extractor-architectures-per-dataset)
   for what CIFAR-10/Imagenette use today and how to pick or justify one for a new
   dataset.
3. **Train the backbone and the feature processor** under a new
   `examples/model_training/scenario_<name>/` folder (`train_core.py` +
   `make_feature_processor.py`), producing an INT8 TFLite feature extractor and a
   `feature_processor.npz`.
4. **Create one `examples/python_examples/scenario_<name>_<classifier>/` folder per
   classifier variant** you want to benchmark (KNN / Nearest Centroid / KMeans), each
   with its own `config.json`, `main.py`, `create_initial_state.py`, and a `model/`
   folder holding the trained backbone + feature processor.

## 2. Benchmark — register and run the scenario

Once a scenario folder trains/loads correctly, the benchmark pipeline picks it up
without any code changes — including packaging it for deployment, which happens
automatically and is not a separate manual step. Full detail, step by step:
[`server/benchmark/README.md` → "Adding A New Scenario"](server/benchmark/README.md#adding-a-new-scenario).
It covers the required folder layout, where test images must live, registering the
scenario name in `experiment_names`, and the `--dry-run` sanity check to run before
deploying anything to the Pi.
