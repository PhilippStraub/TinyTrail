# Client Files

This includes all client components of the framework: (i) Deployment Executor, (ii) Drift Detection Library, (iii) Metric and Drift Transmitter

## Component Overview

### Deployment Executor
This core component includes all files required for executing the inference pipeline, e.g., the initial state of the KNN, a configuration file for the drift-detection library, or the feature extractor as a TFLite model. To integrate the drift detection library into the source code, a feature extractor and a KNN are required. Automatic deployment execution requires a \texttt{main.py} file, while static components, such as the drift-detection library and rover steering controls, are installed beforehand and only need to be imported.

### Drift Detection Library
Using feature encodings inside the KNN, this component autonomously conducts unsupervised concept drift detection and metric collection for each inference.  

A **concept drift** is manifested as a change in $P(X)$ or $P(Y|X)$, when comparing new data to an established reference distribution. To detect this change, the inference pipeline works as follows:

1. Incoming raw sensor data is transformed by the feature extractor into a high-dimensional feature vector.  
2. To this vector, L2-normalization, z-score standardization, and PCA are applied to enhance the robustness of distance-based metrics.  
3. The reduced vector allows the KNN to provide predictions and Euclidean distances (cf. Equation \ref{eq:Euclidean_distances}) to the $k$-nearest neighbors for each query point.  
4. A proxy metric $x_t$ is created based on the average of these $k$ distances (cf. Equation \ref{eq:KNN_distance}), whereby a statistically significant, sustained variation in $x_t$ suggests that new data is dissimilar to the reference data.  
5. To detect these statistically significant changes, a two-sided Page-Hinkley test is employed.

---

#### Euclidean Distance

The Euclidean distance to a neighbor $\mathbf{r}_j$ for a point $\mathbf{y}_t$ is:

$$
d(\mathbf{y}_t,\mathbf{r}_j) = \sqrt{\sum_{i=1}^K \bigl(y_{t,i} - r_{j,i}\bigr)^2}
\tag{1}\label{eq:Euclidean_distances}
$$

leading to the average KNN distance (proxy metric):

$$
x_t = \frac{1}{k}\sum_{j=1}^k d(\mathbf{y}_t,\mathbf{r}_j)
\tag{2}\label{eq:KNN_distance}
$$

---

#### Page-Hinkley Test

The **Page-Hinkley test** detects statistically significant deviations based on a reference mean $\mu_{\text{ref}}$.  
This reference mean $\mu_{\text{ref}}$ can either be collected during a warm-up phase or provided pre-computed based on a curated and balanced training dataset.  

For our experiments, the latter is used and computed during the initialization of the KNN. For this, a **Leave-One-Out Cross-Validation** scheme is applied:  
For each feature vector, the distance to all other training samples is computed, and the $k$ nearest distances are taken.  
Resulting in a distribution of mean distances, the reference mean and standard deviation can be determined and saved within a JSON file, loaded at runtime.  

Using the reference mean $\mu_{\text{ref}}$, for each new data point $x_t$, the deviation $\Delta_t$ is calculated:

$$
\Delta_t = x_t - \mu_{\text{ref}}
\tag{3}
$$

Two cumulative sums are used to accumulate evidence for a change in data:  
$S_{\text{up}}$ for positive drifts (indicating unexpectedly large distances) and  
$S_{\text{down}}$ for negative drifts (indicating unexpectedly small distances).  

If either of the sums becomes negative, it is reset to zero, thereby forgetting data that does not provide evidence for a change.

$$
S_{\text{up}, t} = \max(0, S_{\text{up}, t-1} + \Delta_t - \delta)
\tag{4}
$$

$$
S_{\text{down}, t} = \max(0, S_{\text{down}, t-1} - \Delta_t - \delta)
\tag{5}
$$

To be flagged as a drift, one of the two cumulative sums must exceed a predefined detection threshold $\lambda$.  
The operational flow for continuous drift detection is detailed in Algorithm \ref{alg:drift_detection}, including hysteresis based on predicted confidence.  

On detection of a drift, the processed image and details required for the execution of Algorithm \ref{alg:drift_detection} are logged.  
Drift-independent, after each inference, a performance monitoring JSON is created, which includes:

- Unix timestamp  
- frame count  
- prediction  
- confidence  
- latency (in milliseconds)  
- CPU utilization  
- memory utilization  
- drift detection status (boolean)  
- drift detection details, if applicable  



### Metric and Drift Transmitter
This component packages, compresses, and sends all metric and drift files to the middleware. Based on the files created by the drift detection library, a compressed ZIP file is created. By using the device's UUID in an analysis stream, the component ensures that the maximum message size of 8 MB is not exceeded and that all packages are delivered to the middleware. After successful transmission, the involved files are deleted.


## Prerequisites
This is a pyhton project, this requires on the Pi Zero 2W installation of:
- Python 3.11
- NumPy 1.24.2
- SSH access to your target device (e.g., `pi@wolfgang.local`)
- The target device is connected to the same network


## Operational flow
1. Execute the deployment-reiver on the Pi zero via SSH
2. Transmit a deployment to NATS: Be sure to check the device_id is set correct inside the script, otherwise you won't receive anything
3. Depoyment is automatically executed, logs are automatically transmitted and the Pi listens for new deployments to execute
4. To stop, press "STRG + C". This terminates the process

## Execution
1. Replicate this client folder including all subfolders to the Pi Zero 2W using scp:
```bash
scp -r /path/to/tinysphere-deployment-pipeline pi@wolfgang:~
```
2. Switch to the created directory "tinysphere-deployment-pipeline" and run:
```bash
python deployment-receiver.py
```
3. Deployment receiver starts up and listens to NATS Jetstream for deployments

