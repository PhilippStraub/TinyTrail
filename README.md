# Twin-Ready On-Device Learning: Selecting ML Model Configurations under Resource Constraints (TinyTrail Repository)

In this repository, the *TinyTrail* tool and results for our paper titled "*Twin-Ready On-Device Learning: Selecting ML Model Configurations under Resource Constraints*" can be accessed. In our paper, we present and validate the tool-supported evaluation process that enables the automatic, empirical evaluation of on-device learning configurations through metric collection and use case-specific suitability estimation, to aid decision-making before digital twin deployment. 
*Currently, the contributed paper is under review.* However, all data supporting the findings of our study as well as the tool can be publicly accessed in this repository. 



### Access the Empirical Data of the Paper

The **replication package** of the paper can be downloaded here: [`replication-package.zip`](replication-package.zip).

The raw empirical data of our paper can be found here: [`server/benchmark/results`](server/benchmark/results).

To access the public dataset of the rover-lab click here: [`Rover dataset`](https://huggingface.co/datasets/Fundulus/rover-lab).

# The TinyTrail Evaluation Process Overview
*Details on the evaluation process can be retrieved from the paper*. A short summary for proper repository usage will be provided on acceptance. 

## Tool Support: TinyTrail 
The evaluation process conducts controlled experiments on the target hardware of the selected use case, based on the user-provided parameterization. To support in the execution of steps that must handle high combinatorics, we developed TinyTrail. The tool supports model training, experimentation, deployment and metric measurement according to process parameters.


### Tool Archicture Overview (standalone TinyTrail execution)


<p align="center">
  <img src="./tinyTrail-detailed.png" alt="TinyTrail Tool Overview" width="600">
</p>

In subsequent folders, components inlcuding setup are explained in detail.


# Adding a new Use Case..

## ..for an Image Classification Task 
The case of our paper conducts image classification on a resource-constrained rover, based on a Pi Zero 2w. If you want to add a new use case, please follow the steps of the evaluation process, i.e., determine a use case-tailored suitability metric. Once defined, the exisiting procedures for data collection and analysis in this repository, can be reused, tailored to your use case's needs. 


## ..for another AI task
This repository builds on the on-device learning approach by Disabato et al. \[1]. Therefore, transferability to other AI-tasks is possible. For this, please follow the steps of the evaluation process, i.e., determine a use case-tailored suitability metric and select appropriate synthetic datasets and feature extractors according to your task. Once defined, the exisiting procedures for data collection and analysis in this repository, can be adapted, tailored to your use case's needs. 


[1] S. Disabato and M. Roveri, "*Incremental On-Device Tiny Machine Learning*," in Proceedings of the 2nd International Workshop on Challenges in Artificial Intelligence and Machine Learning for Internet of Things (SenSys '20), ACM, Nov. 2020, pp. 7–13. https://doi.org/10.1145/3417313.3429378

## Overview
Adding a new use case means adding a new **datasets** end-to-end: including a trained feature extractor, one or more ML model configurations built on it, and a benchmark entry that measures it on the Pi Zero 2W. Every dataset already in the repo (CIFAR-10,
Imagenette, MNIST, Rover-Lab) was built this same way. The steps below only order and abstract that recipe — the actual detail lives in the two component READMEs linked from each step. Before starting, please read them too.

### 1. Retrainer — dataset, feature extractor, ML model configuration

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
4. **Create one `examples/python_examples/scenario_<name>_<configuration>/` folder per
   configuration variant** you want to benchmark (KNN, Nearest Centroid, KMeans each with and without PCA), each with its own `config.json`, `main.py`, `create_initial_state.py`, and a `model/` folder holding the trained backbone + feature processor.

### 2. Benchmark — register and run the scenario

Once a scenario folder trains/loads correctly, the benchmark pipeline picks it up
without any code changes — including packaging it for deployment, which happens
automatically and is not a separate manual step. Full detail, step by step:
[`server/benchmark/README.md` → "Adding A New Scenario"](server/benchmark/README.md#adding-a-new-scenario).
It covers the required folder layout, where test images must live, registering the
scenario name in `experiment_names`, and the `--dry-run` sanity check to run before
deploying anything to the Pi.



## Reference this repository 
*On acceptance, a paper reference will be provided here.*
