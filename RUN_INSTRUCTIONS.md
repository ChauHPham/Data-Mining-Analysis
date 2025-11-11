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

