# Multi-Scenario Benchmark Findings

**Project:** Applied Machine Learning, DBE SS26 - KNN/reference-sample sweet spot in the
TinyMLOps framework.
**Hardware:** Raspberry Pi Zero 2W with external power logging via Voltcraft DSP-3005 at the
5 V rail.
**Run:** 1440 completed experiments: 6 scenarios, 2 PCA modes, 12 sample sizes, 10 seeds.

Final generated export sets:

- `exports/cifar10/analysis_20260705_193251/`
- `exports/imagenet/analysis_20260708_140042/` (thread-pinned re-run; supersedes the earlier confounded Imagenette runs)

The exports contain the full tables, figures, generated findings, and the project-requirements
checklists used for the write-up.

## Headline Answers

1. **Reference-sample sweet spot:** the practical knee is early. CIFAR-10 reaches about 95% of
   its best mean accuracy at 10-20 samples/class. Imagenette reaches that threshold at 5-10
   samples/class.
2. **Classifier comparison:** Nearest Centroid and KMeans are competitive with KNN on CIFAR-10;
   on Imagenette, KNN and Nearest Centroid are the strongest observed classifiers.
3. **PCA trade-off:** CIFAR-10 `pca95var` keeps 95.14% variance and slightly improves accuracy.
   Imagenette `pca95var` keeps 95.02% variance, but its measured energy cost is notably higher,
   so the PCA choice should be argued as a dataset-specific accuracy/memory/energy trade-off.

## Scenario Coverage

| Dataset | Scenarios | Runs | PCA modes | Samples/class | Seeds |
|---|---|---:|---|---|---:|
| CIFAR-10 | KNN, Nearest Centroid, KMeans | 720 | `nopca`, `pca95var` | 2..100 | 10 |
| Imagenette | KNN, Nearest Centroid, KMeans | 720 | `nopca`, `pca95var` | 2..100 | 10 |

## Accuracy Knees

Smallest `samples_per_class` reaching 95% of that scenario/PCA pipeline's best mean accuracy:

| Scenario | Classifier | nopca | pca95var |
|---|---|---:|---:|
| CIFAR-10 KMeans | KMeans | 10 | 10 |
| CIFAR-10 KNN | KNN | 20 | 10 |
| CIFAR-10 Nearest Centroid | Nearest Centroid | 10 | 10 |
| Imagenette KMeans | KMeans | 10 | 10 |
| Imagenette KNN | KNN | 10 | 10 |
| Imagenette Nearest Centroid | Nearest Centroid | 5 | 5 |

## Best Mean Accuracy

| Dataset | Classifier | PCA | Best samples/class | Mean accuracy |
|---|---|---|---:|---:|
| CIFAR-10 | KMeans | `pca95var` | 100 | 0.8070 |
| CIFAR-10 | KNN | `pca95var` | 90 | 0.7935 |
| CIFAR-10 | Nearest Centroid | `pca95var` | 100 | 0.8095 |
| Imagenette | KMeans | `pca95var` | 100 | 0.9730 |
| Imagenette | KNN | `pca95var` | 100 | 0.9835 |
| Imagenette | Nearest Centroid | `pca95var` | 90 | 0.9820 |

## PCA Effect Summary

| Dataset | Classifier | Mean accuracy diff, PCA - no PCA | Wilcoxon p |
|---|---|---:|---:|
| CIFAR-10 | KMeans | +0.00875 | 0.00009 |
| CIFAR-10 | KNN | +0.00600 | 0.00020 |
| CIFAR-10 | Nearest Centroid | +0.00662 | 0.00001 |
| Imagenette | KMeans | +0.00167 | 0.30055 |
| Imagenette | KNN | -0.00196 | 0.36482 |
| Imagenette | Nearest Centroid | +0.00142 | 0.03143 |

## Energy Snapshot

> **⚠️ The Imagenette energy figures below are from the confounded multi-threaded run.** Their high
> `pca95var` cost is an OpenBLAS/numpy multi-threading artifact, not an intrinsic PCA cost. For the
> write-up use the thread-pinned re-run `exports/imagenet/analysis_20260708_140042/` and regenerate
> this table from it. The CIFAR-10 figures are unaffected.

Net charge per approximately 200-image run, averaged by classifier/PCA:

| Dataset | Classifier | nopca mAh/run | pca95var mAh/run |
|---|---|---:|---:|
| CIFAR-10 | KMeans | 0.4554 | 0.4649 |
| CIFAR-10 | KNN | 0.5072 | 0.5003 |
| CIFAR-10 | Nearest Centroid | 0.4525 | 0.4990 |
| Imagenette | KMeans | 3.2857 | 7.3963 |
| Imagenette | KNN | 3.4307 | 7.4774 |
| Imagenette | Nearest Centroid | 3.2422 | 7.5971 |

## Use In The Write-Up

- Use the generated `findings_*.md` files for detailed dataset-specific tables and figure paths.
- Use the `project_requirements_*.csv` files as a checklist against the assignment PDF.
- Keep raw `results/` outside normal Git unless the team explicitly wants a large reproducibility
  archive.
