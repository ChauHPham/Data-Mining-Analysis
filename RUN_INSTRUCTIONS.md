# How to Run the Code

## Prerequisites

Make sure you have the required Python packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 1. Cluster Analysis

### Quick Test (Smaller k range for faster testing)
```bash
python cluster_analysis.py --min-k 2 --max-k 4 --pca-components 30
```

### Full Run (Default settings - Tests k=2 to k=6, 50 PCA components)
```bash
python cluster_analysis.py
```

### 3D Visualization (instead of 2D)
```bash
python cluster_analysis.py --vis-dims 3
```

### Custom Run
```bash
python cluster_analysis.py \
  --data data/diabetic_data.csv \
  --min-k 2 \
  --max-k 10 \
  --pca-components 100 \
  --vis-dims 2 \
  --random-seed 42
```

**Parameters:**
- `--data`: Path to CSV data file (default: data/diabetic_data.csv)
- `--min-k`: Minimum number of clusters to test (default: 2)
- `--max-k`: Maximum number of clusters to test (default: 6)
- `--pca-components`: Number of PCA components for clustering (default: 50)
- `--vis-dims`: Choose 2 or 3 for 2D/3D scatter plot visualization (default: 2)
- `--random-seed`: Random seed for reproducibility (default: 42)

**Optimizations Already in Place:**
- ✅ PCA reduces 2,389 features → 50 components
- ✅ Silhouette computed on 5,000 sampled points (not all 101,767!)
- ✅ KMeans++ initialization (fast & accurate)
- ✅ Vectorized distance computation

**Expected Output:**
- Console output showing silhouette scores for each k
- `silhouette_scores.png` - Line plot of silhouette scores vs k (KMeans++)
- `clustering_k*.png` - Best clustering visualization (KMeans++)
- `readmission_labels_pca.png` - Optional comparison with actual labels (2D only)

**Expected Runtime:** 
- Quick test (k=2-4, 30 PCA): ~20 seconds
- Full run (k=2-6, 50 PCA): ~30-60 seconds
- Extended run (k=2-10, 100 PCA): ~2-3 minutes

---

## 2. Outlier Detection

### Quick Test (Default settings - 1% contamination)
```bash
python outlier_detection.py
```

### More Sensitive Detection (5% expected outliers)
```bash
python outlier_detection.py --contamination 0.05
```

### Adjust LOF Sensitivity
```bash
python outlier_detection.py --lof-neighbors 10 --contamination 0.02
```

### Custom Run
```bash
python outlier_detection.py \
  --data data/diabetic_data.csv \
  --contamination 0.01 \
  --lof-neighbors 20 \
  --output-dir outlier_plots \
  --random-state 42
```

**Parameters:**
- `--data`: Path to CSV data file (default: data/diabetic_data.csv)
- `--contamination`: Expected proportion of outliers, e.g., 0.01 = 1% (default: 0.01)
- `--lof-neighbors`: Number of neighbors for LOF algorithm (default: 20)
- `--output-dir`: Directory to save plots (default: outlier_plots/)
- `--random-state`: Random seed for reproducibility (default: 42)

**What It Does:**
1. **Isolation Forest**: Isolates outliers by random partitioning
2. **Local Outlier Factor (LOF)**: Detects outliers based on local density

**Expected Output:**
- Console output with:
  - Number of outliers detected by each method
  - Score ranges and top outlier scores
  - Overlap analysis (common outliers across methods)
  - Recommendations for handling outliers
- `outliers_comparison.png` - Side-by-side comparison of both methods
- `outliers_isolation_forest.png` - Detailed Isolation Forest visualization
- `outliers_lof.png` - Detailed LOF visualization

**Expected Runtime:** 
- Default run (1% contamination): ~1-2 minutes
- Higher contamination (5%): ~1-2 minutes

**Interpretation:**
- **Common outliers** (detected by both methods): High-confidence anomalies
- **Method-specific outliers**: May be noise or specific to that algorithm
- **Next steps**: Review outliers to decide if they are:
  - Data errors/noise → Remove from dataset
  - Valid rare events → Keep or handle specially

---

## 3. Classification Analysis

### Quick Test (Fewer trees for faster testing)
```bash
python classification_analysis.py --n-estimators 10 --lasso-C 0.5
```

### Full Run (Default: Lasso feature selection, 20 trees, depth 10)
```bash
python classification_analysis.py
```

### High Accuracy Run (More trees and depth)
```bash
python classification_analysis.py --n-estimators 50 --lasso-C 0.5 --max-depth 15
```

### With Lasso (L1) Feature Selection (Default)
```bash
# Use Lasso with default C=0.1 - 3-7 minutes
python classification_analysis.py

# More aggressive feature selection (smaller C = fewer features)
python classification_analysis.py --lasso-C 0.01

# Less aggressive feature selection (larger C = more features)
python classification_analysis.py --lasso-C 1.0

# Without feature selection (uses all features, slower)
python classification_analysis.py --feature-selection none
```

### With Hyperparameter Tuning (Recommended for Best Performance)
```bash
# Quick tuning (10 iterations) - 5-10 minutes
python classification_analysis.py --tune-hyperparameters --tuning-iterations 10

# Standard tuning (20 iterations) - 10-20 minutes
python classification_analysis.py --tune-hyperparameters --tuning-iterations 20

# Extensive tuning (50 iterations) - 25-40 minutes
python classification_analysis.py --tune-hyperparameters --tuning-iterations 50 --n-features 200

# Combine Lasso feature selection with tuning
python classification_analysis.py --feature-selection lasso --tune-hyperparameters --tuning-iterations 20
```

### Custom Run
```bash
python classification_analysis.py \
  --data data/diabetic_data.csv \
  --test-size 0.2 \
  --n-estimators 50 \
  --max-depth 15 \
  --min-samples-split 2 \
  --criterion gini \
  --max-features sqrt \
  --cv-folds 5 \
  --random-seed 42
```

**Parameters:**
- **`--feature-selection METHOD`**: Feature selection method: lasso or none (default: **lasso**)
- **`--lasso-C FLOAT`**: Inverse regularization for Lasso (default: 0.1, smaller = fewer features)
- `--n-estimators`: Number of trees (default: 20)
- `--max-depth`: Max tree depth (default: 10)
- `--cv-folds`: Cross-validation folds: 5 or 10 (default: 5)
- `--criterion`: Split criterion: gini or entropy (default: gini)
- `--max-features`: Features per split: sqrt, log2, or int (default: sqrt)
- `--test-size`: Test set proportion (default: 0.2)
- **`--tune-hyperparameters`**: Enable hyperparameter tuning (flag)
- **`--tuning-iterations N`**: Number of RandomizedSearchCV iterations (default: 20)

**Feature Selection:**
- **Lasso (L1) Regularization** (default, per project requirements):
  - ✅ Embedded method using Logistic Regression with L1 penalty
  - ✅ Automatically zeros out less important feature coefficients
  - ✅ Tends to select fewer features (more aggressive)
  - ✅ Better for handling correlated features
  - ✅ Adjust `--lasso-C` to control sparsity (smaller C = fewer features)
  - Typically selects 40-100 features from 2,389 (depending on C)
  - **Speed Improvement:** ~50-80x faster training!

- **No Feature Selection** (optional):
  - Use all 2,389 features
  - Slower but allows comparison
  - Use with: `--feature-selection none`

**Expected Output:**
- Console output showing:
  - **(If Lasso enabled)** Feature selection results with L1 coefficients
  - Training/test accuracy, precision, recall, F1-score
  - **AUC-ROC scores for both train and test sets**
  - Cross-validation scores across folds
  - Detailed classification reports
  - **(If tuning enabled)** Best hyperparameters found
  - **(If tuning enabled)** Before/After comparison and improvement analysis
- Visualization files:
  - `confusion_matrix_test.png` - Confusion matrix for test set
  - `confusion_matrix_train.png` - Confusion matrix for training set
  - `roc_curve_test.png` - **ROC curve for test set with AUC**
  - `roc_curve_train.png` - **ROC curve for training set with AUC**
  - `cv_scores.png` - Cross-validation scores across folds
- **(If tuning enabled)** Additional files:
  - `confusion_matrix_tuned.png` - Confusion matrix for tuned model
  - `roc_curve_tuned.png` - ROC curve for tuned model
  - `model_comparison.png` - **Side-by-side comparison chart**

**Expected Runtime:** 
- Quick test (10 trees, 50 features): ~1-2 minutes
- Full run (20 trees, 100 features, depth 10): ~3-7 minutes
- High accuracy (50 trees, 200 features, depth 15): ~15-25 minutes
- **With hyperparameter tuning (10 iterations)**: ~5-10 minutes
- **With hyperparameter tuning (20 iterations)**: ~10-20 minutes
- **With hyperparameter tuning (50 iterations)**: ~25-40 minutes

---

Here you go — **fully formatted RUN_INSTRUCTIONS** entries for your four scripts
(**Hierarchical Clustering**, **Elliptic Envelope**, **k-NN Classifier**, and **k-NN Grid Search**)
written *in the exact style and structure* of the examples you provided.

You can paste these directly into `RUN_INSTRUCTIONS.md`.

---

# **4. Hierarchical Clustering**

Performs **Agglomerative Hierarchical Clustering** using your custom `HierarchicalClustering` class with PCA dimensionality reduction.

---

## **Quick Test (Smaller sample size for speed)**

```bash
python hierarchical_clustering_analysis.py --sample-size 1000 --pca-components 30
```

## **Full Run (Default: 2,000 samples, 50 PCA components)**

```bash
python hierarchical_clustering_analysis.py
```

## **Custom Run**

```bash
python hierarchical_clustering_analysis.py \
  --data data/diabetic_data.csv \
  --sample-size 3000 \
  --pca-components 50 \
  --min-k 2 \
  --max-k 6 \
  --random-state 42
```

### **Parameters**

* `--data`: Path to dataset (default: data/diabetic_data.csv)
* `--sample-size`: Number of rows to sample for clustering (default: 2000)
* `--pca-components`: Number of PCA components before clustering (default: 50)
* `--min-k`: Minimum clusters tested (default: 2)
* `--max-k`: Maximum clusters tested (default: 6)
* `--random-state`: Seed for reproducibility (default: 42)

### **Expected Output**

* Silhouette scores for k=2…6
* `hierarchical_silhouette_scores.png`
* `hierarchical_clustering_k*.png` (best k)
* 2D PCA cluster visualization

### **Runtime**

* ~20–40 seconds (default settings)

---

# **5. Elliptic Envelope Outlier Detection**

Runs your custom **Mahalanobis Elliptic Envelope** model with PCA preprocessing and 2D visualizations.

---

## **Quick Test (Default: contamination=0.05, sample=5000)**

```bash
python elliptic_envelope.py
```

## **More Sensitive Detection (Higher contamination rate)**

```bash
python elliptic_envelope.py --contamination 0.10
```

## **Custom Run**

```bash
python elliptic_envelope.py \
  --data data/diabetic_data.csv \
  --contamination 0.03 \
  --sample-size 4000 \
  --pca-components 50 \
  --output-dir elliptic_outlier_plots \
  --random-state 42
```

### **Parameters**

* `--data`: CSV path (default: data/diabetic_data.csv)
* `--contamination`: Expected proportion of anomalies (default: 0.05)
* `--sample-size`: Number of rows to sample (default: 5000)
* `--pca-components`: PCA components before envelope fitting (default: 50)
* `--output-dir`: Where to save plots
* `--random-state`: Random seed

### **Expected Output**

* Console summary of:

  * Outlier count
  * Score ranges
  * Explained variance from PCA
* `elliptic_outliers_pca.png`
* `elliptic_scores_hist.png`

### **Runtime**

* ~20–30 seconds

---

# **6. k-NN Classifier (with PCA + Batched KNN)**

Runs your custom **memory-safe batched KNN classifier** with 50-dim PCA preprocessing.

---

## **Quick Test (Default settings: k=5, Euclidean, uniform)**

```bash
python knn_classifier.py
```

## **Try Manhattan Distance**

```bash
python knn_classifier.py --p 1
```

## **Weighted Distance Voting**

```bash
python knn_classifier.py --weights distance
```

## **Custom Run**

```bash
python knn_classifier.py \
  --data data/diabetic_data.csv \
  --test-size 0.2 \
  --n-neighbors 7 \
  --weights uniform \
  --p 2 \
  --output-dir knn_results \
  --random-state 42
```

### **Parameters**

* `--data`: CSV path (default: data/diabetic_data.csv)
* `--test-size`: Test split proportion (default: 0.2)
* `--n-neighbors`: Number of neighbors (default: 5)
* `--weights`: `uniform` or `distance`
* `--p`: Minkowski p → `1` (Manhattan) or `2` (Euclidean)
* `--output-dir`: Save figures
* `--random-state`: Random seed

### **Expected Output**

* Accuracy, precision, recall, F1
* `knn_confusion_matrix.png`
* `knn_pca_predictions.png`

### **Runtime**

* ~1–2 minutes

---

# **7. k-NN Grid Search (K-fold CV on PCA space)**

Performs manual **grid search** over k, weights, and p with cross-validation on a PCA-reduced training subset.

---

## **Quick Test (k=3,5,7; cv=3; PCA=50)**

```bash
python knn_grid_search.py
```

## **Try Manhattan Distance as well**

```bash
python knn_grid_search.py --p-values 1,2
```

## **Custom Run**

```bash
python knn_grid_search.py \
  --data data/diabetic_data.csv \
  --k-range 3,5,7,9 \
  --weights uniform,distance \
  --p-values 2 \
  --cv 5 \
  --pca-components 50 \
  --max-train-samples 60000 \
  --output-dir knn_grid_results \
  --random-state 42
```

### **Parameters**

* `--k-range`: List of k values (default: 3,5,7)
* `--weights`: `uniform` or `distance`
* `--p-values`: 1 or 2
* `--cv`: Number of folds (default: 3)
* `--pca-components`: PCA dim before grid search (default: 50)
* `--max-train-samples`: Cap training samples for speed
* `--output-dir`: Save CSV + plot
* `--random-state`: Random seed

### **Expected Output**

* Mean CV accuracies for each hyperparameter combination
* Best parameters + CV score
* Final test accuracy
* `knn_grid_results.csv`
* `knn_grid_search_plot.png`

### **Runtime**

* ~2–5 minutes depending on max-train-samples

---


## Quick Verification

### Test if imports work:
```bash
python -c "from kmeans import KMeans; from random_forest import RandomForest; from sklearn.ensemble import IsolationForest; print('Imports OK')"
```

### Test cluster analysis (very quick, k=2 only):
```bash
python cluster_analysis.py --min-k 2 --max-k 2 --pca-components 10
```

### Test outlier detection (very quick, 0.5% contamination):
```bash
python outlier_detection.py --contamination 0.005
```

### Test classification (very quick, 5 trees):
```bash
python classification_analysis.py --n-estimators 5 --cv-folds 3
```

---

## Troubleshooting

### If you get import errors:
```bash
# Make sure you're in the project directory
cd /Users/hoang/CMPT-459-Course-Project

# Check if files exist
ls -la *.py
```

### If you get file not found errors:
```bash
# Check if data file exists
ls -la data/diabetic_data.csv
```

### If you get memory errors:
- Reduce `--pca-components` (e.g., 50 instead of 100)
- Reduce `--n-estimators` (e.g., 50 instead of 100)
- Reduce `--max-k` range (e.g., 2-5 instead of 2-10)

