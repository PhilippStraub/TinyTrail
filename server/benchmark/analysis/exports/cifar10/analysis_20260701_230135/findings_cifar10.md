# CIFAR-10 Benchmark Findings

- Completed experiments analyzed: 720
- Scenarios: scenario_cifar10_kmeans, scenario_cifar10_knn_benchmark, scenario_cifar10_nearest_centroid
- Export folder: `exports\cifar10\analysis_20260701_230135`

## Project/PDF requirements coverage
| pdf_requirement | notebook_documentation |
| --- | --- |
| Hardware | Raspberry Pi Zero 2W with external power supply / Voltcraft DSP-3005 at 5 V; power logs are integrated per experiment. |
| Model task | Image classification benchmark on novelty/test images. |
| Used image dataset(s) | cifar10 |
| Used models / inference pipelines | KMeans, KNN, Nearest Centroid |
| PCA and no-PCA variants | nopca, pca95var |
| Reference samples sweep | 2..100 samples/class; values: [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] |
| Experimental metrics collected | Accuracy, RSS memory, analytical classifier/reference-store memory, CPU, latency, gross/net energy and power. |
| How metrics are collected | Accuracy/latency/CPU/RSS from merged/resource logs; power/energy from external supply metadata; net values subtract idle baseline. |
| Training process | Scenario pipelines are trained on the full training split; benchmark varies only initialization sample size. |
| Seeds | Initialization/test-order seeds present: [42, 123, 456, 789, 1337, 1618, 2021, 2718, 3141, 9876] |
| Train/test size | Median benchmark test images per run: 200; training split is the scenario training set used by the TinyMLOps retrainer. |
| PCA explained variance | pca95var: 95.14% |
| Averaging/statistical tests | Means, std and 95% CI across seeds; paired Wilcoxon PCA vs no-PCA where matched pairs exist. |
| Expected graphs per dataset | Combined grid plus split grids with PCA and without PCA; power distribution boxplot integrated. |

## Scenario coverage
| scenario_name | classifier_label | dataset | runs | seeds | sample_min | sample_max | pca_modes | n_images_median | resolution | raw_dim | feature_dim_min | feature_dim_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_cifar10_kmeans | KMeans | cifar10 | 240 | 10 | 2 | 100 | nopca, pca95var | 200.0 | 32x32 | 160.0 | 31 | 160 |
| scenario_cifar10_knn_benchmark | KNN | cifar10 | 240 | 10 | 2 | 100 | nopca, pca95var | 200.0 | 32x32 | 160.0 | 31 | 160 |
| scenario_cifar10_nearest_centroid | Nearest Centroid | cifar10 | 240 | 10 | 2 | 100 | nopca, pca95var | 200.0 | 32x32 | 160.0 | 31 | 160 |

## Project alignment
- The analysis follows `01 knn-sweet-spot-1.pdf`: identify the KNN/reference-sample sweet spot, document PCA usefulness, and compare KNN with sample-based alternatives under TinyMLOps resource constraints.
- Findings are reported per scenario/classifier so KNN, Nearest Centroid, and KMeans are not averaged into one ambiguous line.
- Power is shown both as net energy/charge summaries and as the required seed-distribution boxplot.
- The figure set includes combined views plus separate with-PCA and without-PCA grids per analyzed dataset family.

## Accuracy knees
| scenario | classifier | pca | max_accuracy | spc_at_95pct | target_accuracy |
| --- | --- | --- | --- | --- | --- |
| scenario_cifar10_kmeans | KMeans | nopca | 0.792 | 10 | 0.7524 |
| scenario_cifar10_kmeans | KMeans | pca95var | 0.807 | 10 | 0.7667 |
| scenario_cifar10_knn_benchmark | KNN | nopca | 0.7885 | 20 | 0.7491 |
| scenario_cifar10_knn_benchmark | KNN | pca95var | 0.7935 | 10 | 0.7538 |
| scenario_cifar10_nearest_centroid | Nearest Centroid | nopca | 0.801 | 10 | 0.7609 |
| scenario_cifar10_nearest_centroid | Nearest Centroid | pca95var | 0.8095 | 10 | 0.769 |

## Best observed mean accuracy per classifier/PCA
| classifier_label | pca_label | samples_per_class | accuracy_mean | accuracy_ci95 |
| --- | --- | --- | --- | --- |
| KMeans | nopca | 90 | 0.792 | 0.0152 |
| KMeans | pca95var | 100 | 0.807 | 0.0108 |
| KNN | nopca | 90 | 0.7885 | 0.0134 |
| KNN | pca95var | 90 | 0.7935 | 0.0097 |
| Nearest Centroid | nopca | 90 | 0.801 | 0.0148 |
| Nearest Centroid | pca95var | 100 | 0.8095 | 0.0163 |

## PCA effect
| scenario | classifier | metric | n_pairs | mean_pca | mean_nopca | mean_diff_pca_minus_nopca | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_cifar10_kmeans | KMeans | accuracy | 120 | 0.77308 | 0.76433 | 0.00875 | 9e-05 |
| scenario_cifar10_kmeans | KMeans | net_energy_per_image_J | 120 | 0.04184 | 0.04098 | 0.00086 | 0.01043 |
| scenario_cifar10_knn_benchmark | KNN | accuracy | 120 | 0.75492 | 0.74892 | 0.006 | 0.0002 |
| scenario_cifar10_knn_benchmark | KNN | net_energy_per_image_J | 118 | 0.04503 | 0.04574 | -0.00071 | 0.05779 |
| scenario_cifar10_nearest_centroid | Nearest Centroid | accuracy | 120 | 0.78925 | 0.78262 | 0.00662 | 1e-05 |
| scenario_cifar10_nearest_centroid | Nearest Centroid | net_energy_per_image_J | 120 | 0.04491 | 0.04072 | 0.00419 | 0.0 |

## Power and resource summary
| classifier_label | pca_label | net_power_W | net_energy_J_img | net_charge_mAh_run | memory_rss_mb | memory_peak_mb | latency_ms | cpu_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KMeans | nopca | 0.1669 | 0.041 | 0.4554 | 59.437 | 59.493 | 43.874 | 4.9013 |
| KMeans | pca95var | 0.1705 | 0.0418 | 0.4649 | 59.4866 | 59.5429 | 43.8972 | 4.9112 |
| KNN | nopca | 0.1838 | 0.0456 | 0.5072 | 61.235 | 61.3882 | 46.5572 | 5.2038 |
| KNN | pca95var | 0.1825 | 0.045 | 0.5003 | 60.0368 | 60.0951 | 45.0902 | 5.0963 |
| Nearest Centroid | nopca | 0.1659 | 0.0407 | 0.4525 | 59.4401 | 59.4978 | 43.9011 | 4.9077 |
| Nearest Centroid | pca95var | 0.1828 | 0.0449 | 0.499 | 59.6003 | 59.6588 | 43.8994 | 4.8798 |

## Figures
- `figures\cifar10_actual_ram_usage.png`
- `figures\cifar10_deep_dive_scenario_cifar10_kmeans_exp_121_nopca_spc2_s42.png`
- `figures\cifar10_power_boxplot_net_avg_power_W.png`
- `figures\cifar10_summary_grid.png`
- `figures\cifar10_with_pca_grid.png`
- `figures\cifar10_without_pca_grid.png`