# CMPT-459-Course-Project

Course project for CMPT 459 - Machine Learning and Data Mining.

This project analyzes diabetic patient data using clustering and classification techniques.

## Dataset

The dataset (`data/diabetic_data.csv`) contains information about diabetic patients, including features such as patient demographics, medical history, medications, and readmission status.

## Report

Report detailing the methodology of the project and any analysis and findings obtained.

[Report Template](https://docs.google.com/document/d/19IGGgtCu5_qCoi9_SZsRXwtrB-Jl6wnWgFtO-hZ9TaM/edit?usp=sharing) (to be filled out)

---

## Project Structure

### 1. Cluster Analysis (K-Means)

Performs optimized K-Means clustering on the diabetic patient dataset to identify patient groups.

**Files:**
- `kmeans.py`: Optimized K-Means implementation with **KMeans++ initialization only**
- `cluster_analysis.py`: Main script for cluster analysis with 2D/3D visualization

**Key Features:**
- ✓ **KMeans++ initialization** for optimal clustering (fast & efficient!)
- ✓ Silhouette score evaluation shown directly in visualizations
- ✓ 2D or 3D scatter plot support
- ✓ Vectorized distance computation for speed
- ✓ Sampling optimization for large datasets
- ✓ **Much faster!** ~5x speed improvement

**Quick Usage:**
```bash
# Quick test (2D visualization) - 20 seconds
python cluster_analysis.py --min-k 2 --max-k 4 --pca-components 30

# Full analysis with 3D visualization - 30-60 seconds
python cluster_analysis.py --vis-dims 3

# Default full run - Tests k=2 to k=6, 50 PCA - takes ~30-60 seconds
python cluster_analysis.py
```

**Parameters:**
- `--data`: Path to CSV data file (default: data/diabetic_data.csv)
- `--min-k N`: Minimum number of clusters (default: 2)
- `--max-k N`: Maximum number of clusters (default: 6)
- `--pca-components N`: PCA components for clustering (default: 50)
- `--vis-dims {2,3}`: 2D or 3D visualization (default: 2)
- `--random-seed N`: Random seed for reproducibility (default: 42)

**Note**: PCA reduces 2,389 features to N components (~2x speedup with 50 vs 100)

**Outputs:**
- `silhouette_scores.png`: Silhouette coefficient vs k (KMeans++)
- `clustering_k*.png`: Best clustering visualization with silhouette score
- `readmission_labels_pca.png`: Optional comparison with actual labels (2D only)

---

### 2. Outlier Detection

Performs outlier detection using two different methods to identify anomalies in the diabetic patient dataset.

**Files:**
- `outlier_detection.py`: Main script implementing both outlier detection methods

**Methods Implemented:**
1. **Isolation Forest**: Ensemble-based method using random partitioning to isolate outliers
2. **Local Outlier Factor (LOF)**: Density-based method comparing local density to neighbors

**Key Features:**
- ✓ Two complementary outlier detection algorithms
- ✓ 2D PCA visualization highlighting outliers
- ✓ Score distribution analysis for each method
- ✓ Overlap analysis to find high-confidence outliers
- ✓ Comparison plots showing both methods side-by-side

**Quick Usage:**
```bash
# Quick test (1% contamination) - 1-2 minutes
python outlier_detection.py

# More sensitive detection (5% contamination)
python outlier_detection.py --contamination 0.05

# Adjust LOF sensitivity
python outlier_detection.py --lof-neighbors 10 --contamination 0.02
```

**Parameters:**
- `--data`: Path to CSV data file (default: data/diabetic_data.csv)
- `--contamination FLOAT`: Expected outlier proportion (default: 0.01 = 1%)
- `--lof-neighbors N`: Number of neighbors for LOF (default: 20)
- `--output-dir PATH`: Directory for plots (default: outlier_plots/)
- `--random-state N`: Random seed for reproducibility (default: 42)

**Outputs:**
- `outliers_comparison.png`: Side-by-side comparison of both methods
- `outliers_isolation_forest.png`: Detailed Isolation Forest results
- `outliers_lof.png`: Detailed LOF results  
- Terminal output with overlap analysis and recommendations

**Interpretation Guide:**
- **Common outliers** (detected by both methods): High-confidence anomalies
- **Method-specific outliers**: May be noise or method-sensitive cases
- **Decision**: Remove if data errors/noise, keep if valid rare events

---

### 3. Classification Analysis (Random Forest)

Performs classification on the diabetic patient dataset using Random Forest with train/test split and cross-validation.

**Files:**
- `node.py`: Node class for decision tree structure
- `decision_tree.py`: Decision Tree implementation with Gini/Entropy criteria
- `random_forest.py`: Random Forest using bootstrap aggregation of Decision Trees
- `classification_analysis.py`: Main classification script

**Key Features:**
- ✓ Train/Test Split: 80% training, 20% testing (configurable)
- ✓ Cross-Validation: 5-fold or 10-fold CV
- ✓ **Feature Selection**: Lasso (L1) Regression - Embedded method with regularization
- ✓ **Hyperparameter Tuning**: RandomizedSearchCV for optimal model selection
- ✓ Comprehensive Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - **AUC-ROC** (Area Under ROC Curve)
- ✓ Visualizations:
  - Confusion matrices (train & test)
  - **ROC curves** (train & test)
  - Cross-validation scores
  - **Before/After tuning comparison** (when tuning enabled)
- ✓ Feature importance analysis and computational efficiency comparison

**Quick Usage:**
```bash
# Quick test (10 trees) - 1-2 minutes
python classification_analysis.py --n-estimators 10 --lasso-C 0.5

# Full analysis with LASSO (default) - 3-7 minutes
python classification_analysis.py

# Adjust Lasso regularization (fewer features)
python classification_analysis.py --lasso-C 0.01

# WITH HYPERPARAMETER TUNING (recommended!) - 10-20 minutes
python classification_analysis.py --tune-hyperparameters --tuning-iterations 20

# Without feature selection (slower, uses all features)
python classification_analysis.py --feature-selection none

# Higher accuracy (50 trees, depth 15) - 15-25 minutes
python classification_analysis.py --n-estimators 50 --lasso-C 0.5 --max-depth 15
```

---

**Parameters:**
- **`--feature-selection METHOD`**: Feature selection method: lasso or none (default: **lasso**)
- **`--lasso-C FLOAT`**: Inverse regularization strength for Lasso (default: 0.1, smaller = more selective)
- `--n-estimators N`: Number of trees in Random Forest (default: 20)
- `--max-depth N`: Maximum depth of each tree (default: 10)
- `--cv-folds N`: Number of cross-validation folds: 5 or 10 (default: 5)
- `--test-size FLOAT`: Test set size (default: 0.2)
- `--criterion`: Split criterion: gini or entropy (default: gini)
- `--max-features`: Features per split: sqrt, log2, or int (default: sqrt)
- **`--tune-hyperparameters`**: Enable hyperparameter tuning (flag, off by default)
- **`--tuning-iterations N`**: Number of RandomizedSearchCV iterations (default: 20)

**Note**: Feature selection reduces 2,389 features to top N (~24x speedup!)

**Outputs:**
- `confusion_matrix_test.png`: Confusion matrix for test set
- `confusion_matrix_train.png`: Confusion matrix for training set
- `roc_curve_test.png`: **ROC curve for test set with AUC score**
- `roc_curve_train.png`: **ROC curve for training set with AUC score**
- `cv_scores.png`: Cross-validation scores across folds

**With Hyperparameter Tuning (--tune-hyperparameters):**
- `confusion_matrix_tuned.png`: Confusion matrix for tuned model
- `roc_curve_tuned.png`: ROC curve for tuned model
- `model_comparison.png`: **Side-by-side comparison of baseline vs tuned model**

---

# **4. Lasso Regression**

L1-regularized linear regression for feature selection and regularization.

**Key Features:**
- ✓ L1 regularization for automatic feature selection
- ✓ Embedded feature selection method
- ✓ Configurable regularization strength (C parameter)

**Usage:**
Used as feature selection in Classification Analysis (see section 3). Reduces 2,389 features to top N selected features.

**Parameters:**
- `--lasso-C FLOAT`: Inverse regularization strength (default: 0.1, smaller = more selective)

---

# **5. Hierarchical Clustering**

Performs **Agglomerative Hierarchical Clustering** on PCA-reduced data and selects the best *k* using silhouette scores.

### **Files**

* `hierarchical_clustering.py`
* `hierarchical_clustering_analysis.py` (notebook-style script)

### **Quick Run**

```bash
python hierarchical_clustering_analysis.py
```

### **What it does**

* Full preprocessing pipeline
* Samples dataset (2,000 rows default)
* PCA (50 components)
* Tests k = 2…6
* Computes silhouette scores
* Produces a 2D PCA scatter plot of the best clustering

### **Outputs**

* `hierarchical_results/`

  * `silhouette_scores.png`
  * `hierarchical_k*_plot.png` (best k)
  * Notebook-style printed analysis

---

# **6. Elliptic Envelope Outlier Detection**

Uses **Mahalanobis-distance Elliptic Envelope** on PCA-reduced features to detect multivariate anomalies.

### **File**

* `elliptic_envelope.py`

### **Quick Run**

```bash
python elliptic_envelope.py
```

### **What it does**

* Full preprocessing pipeline
* Samples 5,000 rows
* PCA (50 components)
* Fits custom Elliptic Envelope
* Identifies top 5% anomalies
* Visualizes outliers in 2D PCA

### **Outputs**

* `elliptic_outlier_plots/`

  * `elliptic_outliers_pca.png`
  * `elliptic_scores_hist.png`
* Terminal summary: # of outliers, score ranges, PCA explained variance

---

# **7. k-NN Classification (Batched Implementation)**

Custom k-NN classifier with **batched distance computation**, PCA-reduced features, and full evaluation.

### **File**

* `knn_classifier.py`

### **Quick Run**

```bash
python knn_classifier.py
```

### **What it does**

* Full preprocessing pipeline
* PCA (50 components)
* Train/test split (80/20)
* Batched Euclidean distance KNN
* Computes:

  * Accuracy
  * Precision
  * Recall
  * F1-score

### **Outputs**

* `knn_results/`

  * `knn_confusion_matrix.png`
  * `knn_pca_predictions.png`
* Terminal performance summary (≈88% accuracy)

---

# **8. k-NN Grid Search (Manual CV Search)**

Runs **manual grid search** for KNN hyperparameters using K-fold cross-validation on PCA features.

### **File**

* `knn_grid_search.py`

### **Quick Run**

```bash
python knn_grid_search.py
```

### **What it does**

* Full preprocessing pipeline
* Train/test split
* PCA (50 components)
* Manual grid search over:

  * k values
  * weights (uniform / distance)
  * Minkowski distance p
* Evaluates best model on hold-out test set

### **Outputs**

* `knn_grid_results/`

  * `knn_grid_search_plot.png`
  * `knn_grid_results.csv`
* Terminal output with:

  * Best hyperparameters
  * Mean CV accuracy
  * Final test accuracy (~0.884)

---

## **9. DBScan Clustering**

Performs **Density-based Clustering** on the PCA-reduced diabetic patient dataset.

### **Files**

* `dbscan_clustering.py`: custom implementation of DBScan
* `dbscan_cluster_analysis.py`: Script that runs and visualizes the clustering. 

### **Input Parameters** 

* `--data`: Path to CSV dataset (default path set to: data/diabetic_data.csv)
* `--pca-components N`: number of PCA components for clustering (default set to: 50)
* `--vis-dims`: 2D or 3D scatter plot visualization (default set to: 2)
* `--sample-size`: Limit number of dataset samples for DBScan running time (default set to: 1000)
* `--random-seed`: Random seed for reproducibility (default set to: 42)

### **Quick Run**

```bash
python dbscan_cluster_analysis.py
```

### **Features**

* Data preprocessed and loaded 
* Takes representative sample of dataset (1000)
* Dimensionality reduced PCA dataset 
* Computes and visualizes silhouette score of clusters 
* Supports a 2D/3D scatterplot of clusterings 

### **Outputs**

* `dbscan.png`
* `dbscan_silhouette_scores.png` 

---

# **10. Support Vector Machine (SVM) Classifier (Enhanced)**

Runs a PCA-assisted Support Vector Machine classifier on the diabetic patient dataset.

### **File**

* `svm_classifier.py`

### **Description**

This script:

* Applies the full preprocessing pipeline
* Reduces dimensionality using PCA (default: 20 components)
* Trains a custom SVM classifier (wrapper around scikit-learn)
* Evaluates:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Produces:

  * Confusion matrix
  * PCA prediction scatter plot

Runtime: **~5 minutes** (training is the slow part).

---

### **Quick Run**

```bash
python svm_classifier.py
```

### **Optional Parameters**

```
--data PATH             Path to dataset (default: data/diabetic_data.csv)
--test-size FLOAT       Test ratio (default: 0.2)
--pca-components N      PCA dimensions (default: 20)
--kernel {linear,rbf,poly,sigmoid}  SVM kernel (default: rbf)
--C FLOAT               Regularization strength (default: 1.0)
--gamma {scale,auto}    Kernel coefficient (default: scale)
--degree N              Degree for polynomial kernel (default: 3)
--random-state N        Random seed (default: 42)
```

---

### **Outputs (saved to `svm_results/`)**

* `svm_confusion_matrix.png`
* `svm_pca_predictions.png`
* Terminal accuracy/precision/recall/F1 summary

---

# **11. SVM Random Search (Manual Hyperparameter Search)**

A fast, manually implemented random search for SVM hyperparameters using PCA-reduced data and K-fold CV (default: **fast mode enabled**).

### **File**

* `svm_random_search.py`

---

### **Quick Run (FAST MODE, recommended)**

```bash
python svm_random_search.py --fast
```

This mode:

* Forces **linear kernel only** (much faster)
* Uses **10 PCA components**
* Subsamples **8,000 training rows**
* Runs **2 random trials**
* CV = 2 folds

Runtime: **~10–20 seconds**.

---

### **Full Run (slow, but more thorough)**

```bash
python svm_random_search.py
```

* Uses full dataset
* Uses full PCA components (default: 50)
* Explores all kernels
* CV defaults to 3 folds
* Runs 10 random trials

Runtime: **20–40 minutes** depending on kernel choices.

---

### **Key Parameters**

```
--data PATH               Path to CSV dataset
--iterations N            Number of random hyperparameter samples (default: 10)
--cv N                    Cross-validation folds (default: 3)
--pca-components N        PCA dimensions (default: 50)
--max-train-samples N     Limit training samples (default: 30000)
--fast                    Enable fast mode (recommended)
--output-dir PATH         Save directory (default: svm_random_results/)
--random-state N          Random seed (default: 42)
```

---

### **Outputs (saved to `svm_random_results/`)**

* `svm_random_results.csv` — all CV trials
* Terminal summary:

  * Best hyperparameters
  * CV accuracy
  * Final test accuracy

Example output:

```
Best params: {'kernel': 'linear', 'C': 7.74, 'gamma': 'auto', 'degree': 3}
Mean CV accuracy: 0.8940
Test accuracy:    0.8884
```

---

## Quick Start

1. **Run Cluster Analysis:**
```bash
python cluster_analysis.py
# Tests k=2 to k=6, 50 PCA components, takes ~30-60 seconds
```

2. **Run Outlier Detection:**
```bash
python outlier_detection.py
# Both methods, 1% contamination, takes ~1-2 minutes
```

3. **Run Classification Analysis:**
```bash
python classification_analysis.py
# 20 trees, 100 features, depth 10, takes ~3-7 minutes
```

4. **Check Results:**
   - Cluster visualizations: `clustering_*.png`
   - Outlier visualizations: `outlier_plots/outliers_*.png`
   - Classification confusion matrices: `confusion_matrix_*.png`
   - ROC curves with AUC: `roc_curve_*.png`
   - Cross-validation scores: `cv_scores.png`

5. **Run Hierarchical Clustering**

```bash
python hierarchical_clustering.py
# Samples 2,000 points, applies PCA (50), computes silhouette scores for k=2..6,
# and generates clustering visualization + silhouette score plot.
```

**Outputs:**

* `hierarchical_silhouette_scores.png`
* `hierarchical_clusters_k*.png`
* `hierarchical_best_clustering.png`

---

6. **Run Elliptic Envelope Outlier Detection**

```bash
python elliptic_envelope.py
# Samples 5,000 points, applies PCA (50), detects ~5% anomalies,
# and generates PCA scatter + score histogram.
```

**Outputs:**

* `elliptic_outlier_plots/elliptic_outliers_pca.png`
* `elliptic_outlier_plots/elliptic_scores_hist.png`

---

7. **Run k-NN Classifier (Batched + PCA)**

```bash
python knn_classifier.py
# PCA (50), train/test split (80/20), batched distance computation,
# prints Accuracy/Precision/Recall/F1 and saves confusion matrix + PCA prediction scatter.
```

**Outputs:**

* `knn_results/knn_confusion_matrix.png`
* `knn_results/knn_pca_predictions.png`

---

8. **Run k-NN Grid Search (Manual CV)**

```bash
python knn_grid_search.py
# Manual grid search over k, weights, and p using PCA (50) + K-fold CV.
# Saves accuracy vs k plot and evaluates best model on a final test set.
```

**Outputs:**

* `knn_grid_results/knn_grid_results.csv`
* `knn_grid_results/knn_grid_search_plot.png`
* Terminal summary of best parameters and final test accuracy.

---

9. **Run DBScan Clustering**

```bash
python dbscan_cluster_analysis.py
# PCA(50) components, samples 1000 points and generates silhouette score and scatter plot
```

**Outputs:**

* `dbscan.png`
* `dbscan_silhouette_scores.png` 

10. **Run SVM Classifier**

```bash
python svm_classifier.py
# Performs on train/test split of (80/20), evaluates classification accuracy/precison/recall/f-score
```

**Outputs:**

* `svm_confusion_matrix.png`
* `svm_pca_predictions.png`

11. **Run SVM Random Search**

```bash
python svm_random_search.py
# Run SVM Random Search (FAST MODE, recommended)
```

---

## Detailed Documentation

See `RUN_INSTRUCTIONS.md` for detailed usage instructions and `IMPROVEMENTS_SUMMARY.md` for information about recent improvements to the cluster analysis.

---

## Project Contributors

CMPT 459 Course Project Team

---

## License

This project is for educational purposes as part of CMPT 459 coursework.
