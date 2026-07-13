# ImageNet / Imagenette Benchmark Findings

- Completed experiments analyzed: 720
- Scenarios: scenario_imagenette_kmeans, scenario_imagenette_knn, scenario_imagenette_nearest_centroid
- Export folder: `exports\imagenet\analysis_20260713_083731`

## Project/PDF requirements coverage
| pdf_requirement | notebook_documentation |
| --- | --- |
| Hardware | Raspberry Pi Zero 2W with external power supply / Voltcraft DSP-3005 at 5 V; power logs are integrated per experiment. |
| Model task | Image classification benchmark on novelty/test images. |
| Used image dataset(s) | imagenette |
| Used models / inference pipelines | KMeans, KNN, Nearest Centroid |
| PCA and no-PCA variants | nopca, pca95var |
| Reference samples sweep | 2..100 samples/class; values: [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] |
| Experimental metrics collected | Accuracy, RSS memory, analytical classifier/reference-store memory, CPU, latency, gross/net energy and power. |
| How metrics are collected | Accuracy/latency/CPU/RSS from merged/resource logs; power/energy from external supply metadata; net values subtract idle baseline. |
| Training process | Scenario pipelines are trained on the full training split; benchmark varies only initialization sample size. |
| Seeds | Initialization/test-order seeds present: [42, 123, 456, 789, 1337, 1618, 2021, 2718, 3141, 9876] |
| Train/test size | Median benchmark test images per run: 200; training split is the scenario training set used by the TinyMLOps retrainer. |
| PCA explained variance | pca95var: 95.02% |
| Averaging/statistical tests | Means, std and 95% CI across seeds; paired Wilcoxon PCA vs no-PCA where matched pairs exist. |
| Expected graphs per dataset | Combined grid plus split grids with PCA and without PCA; power distribution boxplot integrated. |

## Scenario coverage
| scenario_name | classifier_label | dataset | runs | seeds | sample_min | sample_max | pca_modes | n_images_median | resolution | raw_dim | feature_dim_min | feature_dim_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_imagenette_kmeans | KMeans | imagenette | 240 | 10 | 2 | 100 | nopca, pca95var | 200.0 | 224x224 | 1280.0 | 608 | 1280 |
| scenario_imagenette_knn | KNN | imagenette | 240 | 10 | 2 | 100 | nopca, pca95var | 200.0 | 224x224 | 1280.0 | 608 | 1280 |
| scenario_imagenette_nearest_centroid | Nearest Centroid | imagenette | 240 | 10 | 2 | 100 | nopca, pca95var | 200.0 | 224x224 | 1280.0 | 608 | 1280 |

## Project alignment
- The analysis follows `01 knn-sweet-spot-1.pdf`: identify the KNN/reference-sample sweet spot, document PCA usefulness, and compare KNN with sample-based alternatives under TinyMLOps resource constraints.
- Findings are reported per scenario/classifier so KNN, Nearest Centroid, and KMeans are not averaged into one ambiguous line.
- Power is shown both as net energy/charge summaries and as the required seed-distribution boxplot.
- The figure set includes combined views plus separate with-PCA and without-PCA grids per analyzed dataset family.

## Accuracy knees
| scenario | classifier | pca | max_accuracy | spc_at_95pct | target_accuracy |
| --- | --- | --- | --- | --- | --- |
| scenario_imagenette_kmeans | KMeans | nopca | 0.9715 | 10 | 0.9229 |
| scenario_imagenette_kmeans | KMeans | pca95var | 0.973 | 10 | 0.9243 |
| scenario_imagenette_knn | KNN | nopca | 0.983 | 10 | 0.9338 |
| scenario_imagenette_knn | KNN | pca95var | 0.9835 | 10 | 0.9343 |
| scenario_imagenette_nearest_centroid | Nearest Centroid | nopca | 0.9795 | 5 | 0.9305 |
| scenario_imagenette_nearest_centroid | Nearest Centroid | pca95var | 0.982 | 5 | 0.9329 |

## Best observed mean accuracy per classifier/PCA
| classifier_label | pca_label | samples_per_class | accuracy_mean | accuracy_ci95 |
| --- | --- | --- | --- | --- |
| KMeans | nopca | 100 | 0.9715 | 0.0109 |
| KMeans | pca95var | 100 | 0.973 | 0.0111 |
| KNN | nopca | 100 | 0.983 | 0.0081 |
| KNN | pca95var | 100 | 0.9835 | 0.0059 |
| Nearest Centroid | nopca | 100 | 0.9795 | 0.0099 |
| Nearest Centroid | pca95var | 90 | 0.982 | 0.0064 |

## PCA effect
| scenario | classifier | metric | n_pairs | mean_pca | mean_nopca | mean_diff_pca_minus_nopca | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_imagenette_kmeans | KMeans | accuracy | 120 | 0.93975 | 0.93808 | 0.00167 | 0.30055 |
| scenario_imagenette_kmeans | KMeans | net_energy_per_image_J | 120 | 0.29232 | 0.29572 | -0.0034 | 0.00036 |
| scenario_imagenette_knn | KNN | accuracy | 120 | 0.9455 | 0.94746 | -0.00196 | 0.36482 |
| scenario_imagenette_knn | KNN | net_energy_per_image_J | 120 | 0.30038 | 0.30876 | -0.00838 | 0.0 |
| scenario_imagenette_nearest_centroid | Nearest Centroid | accuracy | 120 | 0.97158 | 0.97017 | 0.00142 | 0.03143 |
| scenario_imagenette_nearest_centroid | Nearest Centroid | net_energy_per_image_J | 119 | 0.29782 | 0.2918 | 0.00602 | 0.0 |

## Power and resource summary
| classifier_label | pca_label | net_power_W | net_energy_J_img | net_charge_mAh_run | memory_rss_mb | memory_peak_mb | latency_ms | cpu_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KMeans | nopca | 0.519 | 0.2957 | 3.2857 | 66.3452 | 67.7277 | 345.729 | 22.1844 |
| KMeans | pca95var | 0.5039 | 0.2923 | 3.248 | 68.7206 | 70.1327 | 356.1317 | 22.1736 |
| KNN | nopca | 0.5222 | 0.3088 | 3.4307 | 79.3821 | 81.8098 | 367.2907 | 22.2153 |
| KNN | pca95var | 0.5089 | 0.3004 | 3.3376 | 75.2118 | 76.8164 | 366.378 | 22.1446 |
| Nearest Centroid | nopca | 0.513 | 0.2918 | 3.2422 | 65.6888 | 67.0732 | 344.8325 | 22.2245 |
| Nearest Centroid | pca95var | 0.5136 | 0.2978 | 3.3087 | 68.3804 | 69.8035 | 355.8638 | 22.1612 |

## Figures
- `figures\imagenet_actual_ram_usage.png`
- `figures\imagenet_deep_dive_scenario_imagenette_kmeans_exp_121_nopca_spc2_s42.png`
- `figures\imagenet_energy_vs_samples.png`
- `figures\imagenet_power_boxplot_net_avg_power_W.png`
- `figures\imagenet_summary_grid.png`
- `figures\imagenet_with_pca_grid.png`
- `figures\imagenet_without_pca_grid.png`