# CMPT-459-Course-Project

Course project for CMPT 459 - Machine Learning and Data Mining.

This project analyzes diabetic patient data using clustering and classification techniques.

## Dataset

The dataset (`data/diabetic_data.csv`) contains information about diabetic patients, including features such as patient demographics, medical history, medications, and readmission status.

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

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
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

---

## Detailed Documentation

See `RUN_INSTRUCTIONS.md` for detailed usage instructions and `IMPROVEMENTS_SUMMARY.md` for information about recent improvements to the cluster analysis.

---

## Project Contributors

CMPT 459 Course Project Team

---

## License

This project is for educational purposes as part of CMPT 459 coursework.
