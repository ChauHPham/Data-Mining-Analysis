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

### 2. Classification Analysis (Random Forest)

Performs classification on the diabetic patient dataset using Random Forest with train/test split and cross-validation.

**Files:**
- `node.py`: Node class for decision tree structure
- `decision_tree.py`: Decision Tree implementation with Gini/Entropy criteria
- `random_forest.py`: Random Forest using bootstrap aggregation of Decision Trees
- `classification_analysis.py`: Main classification script

**Key Features:**
- Train/Test Split: 80% training, 20% testing (configurable)
- Cross-Validation: 5-fold or 10-fold CV
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization
- Feature importance analysis

**Quick Usage:**
```bash
# Quick test (10 trees, 50 features) - 1-2 minutes
python classification_analysis.py --n-estimators 10 --n-features 50

# Full analysis (20 trees, 100 features, depth 10) - 3-7 minutes
python classification_analysis.py

# Higher accuracy (50 trees, 200 features, depth 15) - 15-25 minutes
python classification_analysis.py --n-estimators 50 --n-features 200 --max-depth 15
```

**Parameters:**
- `--n-features N`: Number of best features to select (default: 100)
- `--n-estimators N`: Number of trees in Random Forest (default: 20)
- `--max-depth N`: Maximum depth of each tree (default: 10)
- `--cv-folds N`: Number of cross-validation folds: 5 or 10 (default: 5)
- `--test-size FLOAT`: Test set size (default: 0.2)
- `--criterion`: Split criterion: gini or entropy (default: gini)
- `--max-features`: Features per split: sqrt, log2, or int (default: sqrt)

**Note**: Feature selection reduces 2,389 features to top N (~24x speedup!)

**Outputs:**
- `confusion_matrix.png`: Confusion matrix heatmap
- `feature_importance.png`: Top 20 most important features
- `cross_validation_scores.png`: CV score distribution

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

2. **Run Classification Analysis:**
```bash
python classification_analysis.py
# 20 trees, 100 features, depth 10, takes ~3-7 minutes
```

3. **Check Results:**
   - Cluster visualizations: `clustering_*.png`
   - Silhouette comparison: `silhouette_comparison.png`
   - Confusion matrix: `confusion_matrix.png`
   - Feature importance: `feature_importance.png`

---

## Detailed Documentation

See `RUN_INSTRUCTIONS.md` for detailed usage instructions and `IMPROVEMENTS_SUMMARY.md` for information about recent improvements to the cluster analysis.

---

## Project Contributors

CMPT 459 Course Project Team

---

## License

This project is for educational purposes as part of CMPT 459 coursework.
