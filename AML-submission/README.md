# AML Submission — Sweet Spot for Distance-Based Methods in the TinyMLOps Framework

This is the guide to the GitLab submission. It maps each item the assignment requires to its
concrete location in this repository, with a focus on where to find the **report sources**,
the **figure-creation / analysis** scripts and data, and the **project source code and
measurement scripts**.

> **Paths.** Links in this file are relative to this `AML-submission/` folder. `../` points to
> the repository root, where the actual framework code and the measurement/analysis pipeline
> live. The report itself is bundled here under [`paper/`](paper/).

---

## Required deliverables → where they are

The assignment asks for two separate parts. Both are contained in this repository:

| # | Required deliverable | Location |
|---|---|---|
| 1 | **Full report incl. all sources** — TeX files, figure-creation scripts, figure data | Report + TeX in [`paper/`](paper/); figure-creation scripts and figure data in [`../server/benchmark/analysis/`](../server/benchmark/analysis/) |
| 2 | **Project source code incl. measurement scripts** | Framework code in [`../client/`](../client/), [`../middleware/`](../middleware/), [`../server/`](../server/); measurement pipeline in [`../server/benchmark/`](../server/benchmark/) |

Because the figure-creation scripts and their input data physically live inside the code tree
(they are part of the measurement/analysis pipeline), they are **not duplicated** here — the
table below points directly to them.

---

## 1. Report and report sources

Everything needed to read or recompile the paper:

| Item | Location |
|---|---|
| **Compiled report (PDF)** | [`paper/ACM_Conference_Template_for_Project_Submission.pdf`](paper/ACM_Conference_Template_for_Project_Submission.pdf) |
| **LaTeX source** | [`paper/ACM_Conference_Template_for_Project_Submission.tex`](paper/ACM_Conference_Template_for_Project_Submission.tex) |
| **Bibliography** | [`paper/references.bib`](paper/references.bib) |
| **Assignment / task description** | [`docs/01 knn-sweet-spot-1.pdf`](docs/01%20knn-sweet-spot-1.pdf) |

The `.tex` file pulls its figures directly from the analysis export folders via `\graphicspath`
(see [§3](#3-figure-creation-and-analysis)), so the report, the plotting scripts, and the figure
data all stay in sync.

---

## 2. Project source code and measurement scripts

### Framework source code

| Component | Location | Contents |
|---|---|---|
| **Client (Raspberry Pi)** | [`../client/`](../client/) | On-device runtime: [`deployment-receiver.py`](../client/deployment-receiver.py), [`transmitter.py`](../client/transmitter.py), the `tinylcm_python/` classifier stack ([`knn.py`](../client/tinylcm_python/knn.py), [`drift_monitor.py`](../client/tinylcm_python/drift_monitor.py), [`feature_extractor.py`](../client/tinylcm_python/feature_extractor.py), [`scaler_pca.py`](../client/tinylcm_python/scaler_pca.py)), plus `rover/` and `utils/` |
| **Middleware** | [`../middleware/`](../middleware/) | NATS messaging config ([`nats.conf`](../middleware/nats.conf)) |
| **Server** | [`../server/`](../server/) | `Analytical-Data-Processor/` (telemetry ingest), `Deployment-Packager/` (builds & ships deployments to the Pi), `Retrainer/` (initial-state / feature-processor creation) |

### Measurement scripts (the empirical study)

All under [`../server/benchmark/`](../server/benchmark/):

| Script | Role |
|---|---|
| [`benchmark_pipeline.py`](../server/benchmark/benchmark_pipeline.py) | Orchestrator — sweeps `classifier × PCA config × samples-per-class × seed`, deploys each config to the Pi over NATS, logs power, and collects results into `results/` |
| [`benchmark_config.json`](../server/benchmark/benchmark_config.json) | Sweep definition (sample sizes, PCA configs, timeouts, device id) |
| [`dsp3005_reader.py`](../server/benchmark/dsp3005_reader.py) | Reads the Voltcraft DSP-3005 bench supply over serial to record the power trace |
| [`config_template.json`](../server/benchmark/config_template.json) | Frozen base Pi configuration, patched per experiment |
| [`analysis/verify_power_coverage.py`](../server/benchmark/analysis/verify_power_coverage.py) | **Validity check** — audits every run: does the power trace fully cover its inference window, are the two clocks causally consistent, are there sampling gaps? Writes one row per run |
| [`analysis/verify_window_alignment.py`](../server/benchmark/analysis/verify_window_alignment.py) | **Validity check** — measures the residual clock offset directly from the power edge, by averaging all runs of a dataset family on their own window start |
| [`README.md`](../server/benchmark/README.md) | How to run the sweep |

### Why these measurements are trustworthy (measurement validity)

Energy is measured *outside* the device: the Voltcraft meter has no clock of its own, so the
power trace is timestamped by the **benchmark PC**, while the inference window that the energy
is integrated over comes from the log written on the **Raspberry Pi**. Two clocks, and no
explicit synchronization step in the pipeline. Rather than assume they agree, the two scripts
above verify it after the fact, from the archived files only — neither needs NTP to be trusted.

| Question | Where it is answered |
|---|---|
| **What is the threat, and what does it mean for the results?** | Report, Section 5.4 *Threat to Validity* → [`paper/`](paper/) |
| **How is it checked, how do I re-run it, what came out?** | [`../server/benchmark/analysis/README.md`](../server/benchmark/analysis/README.md) **§10** |
| **Full audit numbers and the error model (German)** | [`../server/benchmark/MESSMETHODIK_ZEITSYNCHRONISATION.md`](../server/benchmark/MESSMETHODIK_ZEITSYNCHRONISATION.md) |
| **Raw evidence, one row per run** | [`../server/benchmark/results/analysis/power_coverage_audit.csv`](../server/benchmark/results/analysis/power_coverage_audit.csv) and [`window_alignment.csv`](../server/benchmark/results/analysis/window_alignment.csv) |

Both CSVs are committed, so the checks are reproducible from this repository alone — `results/`
is otherwise excluded from version control because it holds the 1440 raw experiment folders.

Three of the 1440 runs are excluded from every aggregate because each received a duplicate of
the preceding experiment's result, which put the wrong seed's accuracy and latency in the row.
The exclusion is explicit in [`EXCLUDED_EXPERIMENTS`](../server/benchmark/analysis/analyze_benchmark.py)
and applies to the notebooks and figures alike; the affected configurations use 9 of 10 seeds
and the reported knees are unchanged. Details in §10.4 of the analysis README.

Raw per-experiment measurement output (one folder per run) lands in
`../server/benchmark/results/<scenario>/exp_*/` — each holds `merged_analysis_*.csv` (per-image
predictions/latency/CPU/RAM), `power_*.csv` (the raw power trace), `resource_usage_*.csv`, and
`experiment_metadata.json`.

---

## 3. Figure creation and analysis

This is where "everything related to figure creation and analysis" lives:
[`../server/benchmark/analysis/`](../server/benchmark/analysis/).

### Scripts and notebooks that produce the figures

| File | Role |
|---|---|
| [`benchmark_analysis_cifar10.ipynb`](../server/benchmark/analysis/benchmark_analysis_cifar10.ipynb) | Interactive analysis of all CIFAR-10 runs → CSVs + figures |
| [`benchmark_analysis_imagenet.ipynb`](../server/benchmark/analysis/benchmark_analysis_imagenet.ipynb) | Interactive analysis of all Imagenette runs → CSVs + figures |
| [`benchmark_analysis.ipynb`](../server/benchmark/analysis/benchmark_analysis.ipynb) | Combined / exploratory notebook |
| [`analyze_benchmark.py`](../server/benchmark/analysis/analyze_benchmark.py) | Headless engine and shared extraction logic (imported by the notebooks; also runnable standalone to regenerate CSVs, figures, and the report) |
| [`make_findings_figures.py`](../server/benchmark/analysis/make_findings_figures.py) | Generates the summary/findings figures |
| [`plot_power_boxplot.py`](../server/benchmark/analysis/plot_power_boxplot.py) | Power boxplot used in the analysis |
| [`README.md`](../server/benchmark/analysis/README.md) | Detailed methodology: what each metric means and how energy/accuracy/memory are computed |
| [`FINDINGS.md`](../server/benchmark/analysis/FINDINGS.md) | Roll-up of the key findings |

### Figure data (inputs to the plots)

The notebooks write each run to a timestamped export folder holding both the **aggregated
tables (CSV)** the figures are drawn from and the **rendered figures (PNG)**:

- [`../server/benchmark/analysis/exports/cifar10/`](../server/benchmark/analysis/exports/cifar10/)
- [`../server/benchmark/analysis/exports/imagenet/`](../server/benchmark/analysis/exports/imagenet/)

Each `analysis_<timestamp>/` folder contains `experiment_summary_*.csv`,
`aggregated_by_scenario_*.csv`, `sweet_spots_*_*.csv`, `pca_effect_*.csv`,
`project_requirements_*.csv`, `findings_*.md`, and a `figures/` subfolder of PNGs.

---

## Reproducing the results end-to-end

1. **Measure** — run the sweep from [`../server/benchmark/`](../server/benchmark/):
   `python benchmark_pipeline.py` (deploys each config to the Pi, logs power, writes `results/`).
2. **Analyze / make figures** — open the dataset notebooks in
   [`../server/benchmark/analysis/`](../server/benchmark/analysis/) and run top-to-bottom, or run
   `python analyze_benchmark.py` headless. This writes a fresh `exports/<dataset>/analysis_<timestamp>/`.
3. **Compile the report** — build
   [`paper/ACM_Conference_Template_for_Project_Submission.tex`](paper/ACM_Conference_Template_for_Project_Submission.tex);
   `\graphicspath` picks up the figures from the export folders automatically.

See [`../server/benchmark/README.md`](../server/benchmark/README.md) and
[`../server/benchmark/analysis/README.md`](../server/benchmark/analysis/README.md) for the full
methodology.
